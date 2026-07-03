"""ALIGNN PhononScore with a stable periodic pair-geometry likelihood.

This branch keeps the existing PhononScore heads and adds a periodic edge MDN
that learns a stable-crystal local distance prior. The MDN head conditions on
atom-pair chemistry embeddings, while the target is the periodic edge distance.
During training the MDN NLL is weighted by phonon stability so strongly
unstable structures do not become positive examples for the geometry prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import dgl
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from alignn.graphs import compute_bond_cosines
from alignn.models.alignn_atomwise import ALIGNNAtomWiseConfig, ALIGNNConv, EdgeGatedGraphConv
from alignn.models.utils import MLPLayer, RBFExpansion

from .phonon_ranker_losses import combine_final_score, mdn_log_probability


def stable_sample_weight(target: Tensor, *, threshold: float = -0.1, tau: float = 0.2) -> Tensor:
    """Return a smooth weight that is high for dynamically stable samples.

    The direction matters: more negative minimum frequencies should receive
    smaller geometry-likelihood weights, so they do not teach the MDN that
    unstable local environments are likely stable-crystal geometries.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    return torch.sigmoid((target.view(-1) - float(threshold)) / float(tau))


def edge_batch_from_graph(graph: dgl.DGLGraph) -> Tensor:
    """Map every edge in a batched DGL graph to its parent graph index."""
    counts = torch.as_tensor(graph.batch_num_edges(), dtype=torch.long, device=graph.device)
    if counts.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=graph.device)
    graph_ids = torch.arange(counts.numel(), dtype=torch.long, device=graph.device)
    return torch.repeat_interleave(graph_ids, counts)


def mean_edges_by_graph(values: Tensor, *, edge_batch: Tensor, num_graphs: int) -> Tensor:
    """Average per-edge values into per-graph values."""
    values = values.view(-1)
    if values.numel() == 0:
        return torch.zeros(num_graphs, dtype=values.dtype, device=values.device)
    sums = torch.zeros(num_graphs, dtype=values.dtype, device=values.device)
    counts = torch.zeros(num_graphs, dtype=values.dtype, device=values.device)
    sums.scatter_add_(0, edge_batch, values)
    counts.scatter_add_(0, edge_batch, torch.ones_like(values))
    return sums / counts.clamp_min(1.0)


def weighted_mdn_nll(edge_nll: Tensor, *, edge_batch: Tensor, sample_weight: Tensor) -> Tensor:
    """Stable-weighted MDN negative log likelihood over periodic edges."""
    edge_nll = edge_nll.view(-1)
    if edge_nll.numel() == 0:
        return edge_nll.sum() * 0.0
    edge_weight = sample_weight.view(-1)[edge_batch]
    denom = edge_weight.sum().clamp_min(1e-8)
    return (edge_nll * edge_weight).sum() / denom


class PairMDNHead(nn.Module):
    """Mixture density head for scalar periodic pair distances."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_gaussians: int,
        *,
        distance_max: float = 8.0,
        sigma_min: float = 1e-3,
    ) -> None:
        super().__init__()
        if n_gaussians <= 0:
            raise ValueError("n_gaussians must be positive")
        self.n_gaussians = int(n_gaussians)
        self.distance_max = float(distance_max)
        self.sigma_min = float(sigma_min)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * self.n_gaussians),
        )

    def forward(self, pair_features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw = self.net(pair_features)
        raw_pi, raw_sigma, raw_mu = raw.chunk(3, dim=-1)
        pi = F.softmax(raw_pi, dim=-1)
        sigma = F.softplus(raw_sigma) + self.sigma_min
        mu = torch.sigmoid(raw_mu) * self.distance_max
        return pi, sigma, mu


@dataclass
class PhononGeomScoreOutput:
    min_freq: Tensor
    threshold_logits: Tensor
    threshold_score: Tensor
    pair_geometry_score: Tensor
    final_score: Tensor
    mdn_pi: Tensor
    mdn_sigma: Tensor
    mdn_mu: Tensor
    edge_distance: Tensor
    edge_logp: Tensor
    edge_batch: Tensor


class ALIGNNPhononScoreGeomMDN(nn.Module):
    """ALIGNN PhononScore plus stable periodic pair geometry likelihood."""

    def __init__(
        self,
        config: ALIGNNAtomWiseConfig,
        *,
        thresholds: Sequence[float] = (-0.001, -0.01, -0.1, -1.0),
        beta: float = 0.6,
        alpha: float = 0.1,
        mdn_hidden_dim: int = 128,
        mdn_gaussians: int = 10,
        mdn_distance_max: float = 8.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.thresholds = tuple(float(v) for v in thresholds)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.mdn_distance_max = float(mdn_distance_max)

        self.atom_embedding = MLPLayer(config.atom_input_features, config.hidden_features)
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=config.triplet_input_features),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.alignn_layers = nn.ModuleList(
            [ALIGNNConv(config.hidden_features, config.hidden_features) for _ in range(config.alignn_layers)]
        )
        self.gcn_layers = nn.ModuleList(
            [EdgeGatedGraphConv(config.hidden_features, config.hidden_features) for _ in range(config.gcn_layers)]
        )
        self.readout = dgl.nn.AvgPooling()

        self.min_freq_head = nn.Linear(config.hidden_features, 1)
        self.threshold_head = nn.Linear(config.hidden_features, len(self.thresholds))
        pair_input_dim = config.hidden_features * 4
        self.pair_mdn_head = PairMDNHead(
            pair_input_dim,
            hidden_dim=mdn_hidden_dim,
            n_gaussians=mdn_gaussians,
            distance_max=mdn_distance_max,
        )

    def load_alignn_atomwise_state(self, state_dict: dict[str, Tensor], strict_encoder: bool = False) -> None:
        """Warm-start shared encoder and min-frequency head from ALIGNNAtomWise."""
        own_state = self.state_dict()
        mapped: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            new_key = "min_freq_head." + key[len("fc.") :] if key.startswith("fc.") else key
            if new_key in own_state and own_state[new_key].shape == value.shape:
                mapped[new_key] = value
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        if strict_encoder:
            allowed_missing = ("threshold_head", "pair_mdn_head", "min_freq_head")
            encoder_missing = [k for k in missing if not k.startswith(allowed_missing)]
            if encoder_missing or unexpected:
                raise RuntimeError(
                    f"Could not load ALIGNN encoder cleanly: missing={encoder_missing}, unexpected={unexpected}"
                )

    def encode(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        lattice: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return graph embedding and edge geometry-likelihood tensors."""
        del lattice
        g = g.local_var()
        lg = lg.local_var()
        x0 = self.atom_embedding(g.ndata["atom_features"])

        src, dst = g.edges()
        src_h = x0[src]
        dst_h = x0[dst]
        pair_features = torch.cat([src_h, dst_h, torch.abs(src_h - dst_h), src_h * dst_h], dim=-1)
        mdn_pi, mdn_sigma, mdn_mu = self.pair_mdn_head(pair_features)

        r = g.edata["r"]
        edge_distance = torch.norm(r, dim=1).clamp(min=0.0, max=self.mdn_distance_max)
        edge_logp = mdn_log_probability(mdn_pi, mdn_sigma, mdn_mu, edge_distance)
        edge_batch = edge_batch_from_graph(g)

        x = x0
        y = self.edge_embedding(torch.norm(r, dim=1))
        if len(self.alignn_layers) > 0:
            lg.ndata["r"] = r
            lg.apply_edges(compute_bond_cosines)
            z = self.angle_embedding(lg.edata["h"])
            for alignn_layer in self.alignn_layers:
                x, y, z = alignn_layer(g, lg, x, y, z)

        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        graph_h = self.readout(g, x)
        return graph_h, mdn_pi, mdn_sigma, mdn_mu, edge_distance, edge_logp, edge_batch

    def forward(self, batch: Sequence[Tensor | dgl.DGLGraph]) -> PhononGeomScoreOutput:
        graph, line_graph, lattice = batch
        graph_h, mdn_pi, mdn_sigma, mdn_mu, edge_distance, edge_logp, edge_batch = self.encode(
            graph,
            line_graph,
            lattice,
        )
        min_freq = self.min_freq_head(graph_h).view(-1)
        threshold_logits = self.threshold_head(graph_h)
        threshold_score = torch.sigmoid(threshold_logits).mean(dim=-1)
        pair_geometry_score = mean_edges_by_graph(edge_logp, edge_batch=edge_batch, num_graphs=graph_h.shape[0])
        final_score = combine_final_score(
            min_freq,
            pair_geometry_score=pair_geometry_score,
            multi_threshold_logits=threshold_logits,
            alpha=self.alpha,
            beta=self.beta,
        )
        return PhononGeomScoreOutput(
            min_freq=min_freq,
            threshold_logits=threshold_logits,
            threshold_score=threshold_score,
            pair_geometry_score=pair_geometry_score,
            final_score=final_score,
            mdn_pi=mdn_pi,
            mdn_sigma=mdn_sigma,
            mdn_mu=mdn_mu,
            edge_distance=edge_distance,
            edge_logp=edge_logp,
            edge_batch=edge_batch,
        )
