"""iSyncTab_AV.py

Audio-video extension of iSyncTab.

This module keeps the iSyncTab NS-PFS sequencing mechanism and Linformer-based
fusion, while replacing the original image/tabular encoders with lightweight
projection adapters for precomputed audio and video token features.

RAVDESS experiment convention used in the accompanying experiments:
    audio tokens: [B, 32, 64]   (log-mel tokens)
    video tokens: [B, 16, 2048] (ResNet-50 frame features)

The model itself is configurable and can be used with other fixed-length
precomputed audio/video token sequences by changing audio_len, video_len,
audio_dim, and video_dim.

Requirements:
    pip install torch linformer scipy numpy
"""

import random
from itertools import zip_longest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from linformer import Linformer
from scipy.optimize import linear_sum_assignment


def set_seed(seed=42, deterministic=True):
    """Set Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


# ======================================================
# GPU KMeans
# ======================================================
@torch.no_grad()
def kmeans_torch(X_points, k, iters=50, tol=1e-4, device=None):
    device = device or X_points.device

    P, D = X_points.shape

    if P <= 0:
        raise ValueError("kmeans_torch received zero points.")

    k = max(1, min(int(k), P))

    idx = torch.randperm(P, device=device)[:k]
    C = X_points[idx].clone()

    prev = None

    for _ in range(iters):
        dists = torch.cdist(X_points, C, p=2)
        labels = dists.argmin(dim=1)

        newC = torch.zeros_like(C)

        for ci in range(k):
            mask = labels == ci

            if mask.any():
                newC[ci] = X_points[mask].mean(dim=0)
            else:
                newC[ci] = X_points[
                    torch.randint(0, P, (1,), device=device)
                ]

        shift = (newC - C).abs().mean().item()
        C = newC

        if prev is not None and abs(prev - shift) < tol:
            break

        prev = shift

    return labels, C


# ======================================================
# NS-PFS
# Fixed:
#   - Hungarian matching for cross-modal cluster pairing
#
# Tunable:
#   - num_clusters
#   - metric
#   - bins
#   - sync_temperature
#   - energy_weight
#   - centroid_weight
#   - pair_order after Hungarian matching
#   - within_cluster_order
# ======================================================
class NSPFS_GPU(nn.Module):
    def __init__(
        self,
        num_clusters=5,
        metric="variance",
        bins=32,
        mi_chunk=128,
        sync_temperature=1.0,
        energy_weight=1.0,
        centroid_weight=1.0,
        pair_order="sync",
        within_cluster_order="metric_desc",
        device=None,
    ):
        super().__init__()

        self.k = int(num_clusters)
        self.metric = str(metric).lower()
        self.bins = int(bins)
        self.mi_chunk = int(mi_chunk)

        self.sync_temperature = float(sync_temperature)
        self.energy_weight = float(energy_weight)
        self.centroid_weight = float(centroid_weight)

        self.pair_order = str(pair_order).lower()
        self.within_cluster_order = str(within_cluster_order).lower()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def _metric_values(self, tokens):
        """
        tokens: (B, L, H)
        returns one scalar metric score per token: (L,)
        """
        metric = self.metric
        B, L, H = tokens.shape

        if metric in ("variance", "var"):
            vals = tokens.var(dim=(0, 2), unbiased=False)

        elif metric in ("energy", "l2_energy"):
            vals = tokens.pow(2).mean(dim=(0, 2))

        elif metric in ("mean_abs", "abs"):
            vals = tokens.abs().mean(dim=(0, 2))

        elif metric in ("euclidean", "l2"):
            vals = tokens.mean(dim=0).norm(p=2, dim=1)

        elif metric in ("manhattan", "l1"):
            vals = tokens.mean(dim=0).norm(p=1, dim=1)

        elif metric in ("cosine", "cos"):
            emb = tokens.mean(dim=0)
            emb = F.normalize(emb, dim=1)
            global_centroid = F.normalize(emb.mean(dim=0), dim=0)
            vals = ((emb @ global_centroid) + 1.0) / 2.0

        elif metric in ("correlation", "corr", "pearson"):
            scalar = tokens.mean(dim=2)
            scalar = scalar - scalar.mean(dim=0, keepdim=True)
            scalar = scalar / scalar.std(
                dim=0,
                unbiased=False,
                keepdim=True
            ).clamp_min(1e-8)

            C = (scalar.T @ scalar) / max(float(B), 1.0)

            if L > 1:
                vals = (C.abs().sum(dim=1) - 1.0).clamp_min(0.0) / float(L - 1)
            else:
                vals = torch.ones(L, device=tokens.device)

        elif metric in ("kl", "kl_divergence", "js", "jensen_shannon", "entropy"):
            scalar = tokens.mean(dim=2)
            vals_list = []

            for j in range(L):
                v = scalar[:, j]
                vmin, vmax = v.min(), v.max()

                if float((vmax - vmin).abs()) < 1e-12:
                    vals_list.append(torch.tensor(0.0, device=tokens.device))
                    continue

                hist = torch.histc(
                    v.float(),
                    bins=self.bins,
                    min=float(vmin),
                    max=float(vmax),
                )
                p = hist / hist.sum().clamp_min(1e-8)
                ent = -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum()
                vals_list.append(ent)

            vals = torch.stack(vals_list)

        else:
            raise ValueError(
                f"Unknown NS-PFS metric: {self.metric}. "
                "Supported: variance, energy, mean_abs, euclidean, manhattan, "
                "cosine, correlation, kl, js."
            )

        vals = torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        return vals

    @staticmethod
    @torch.no_grad()
    def _cluster_energy_and_centroid(tokens, idx, metric_vals):
        if idx.numel() == 0:
            return None, None, None

        X = tokens[:, idx, :]

        energy = metric_vals[idx].mean().clamp_min(0.0)

        centroid = X.mean(dim=(0, 1))
        centroid = F.normalize(centroid, dim=0)

        size = int(idx.numel())

        return energy, centroid, size

    @staticmethod
    @torch.no_grad()
    def _order_preserving_unique_complete(seq, total_len, device):
        seq = seq.detach().to(device).long().flatten()

        seen = torch.zeros(total_len, dtype=torch.bool, device=device)
        kept = []

        for idx in seq.tolist():
            if 0 <= idx < total_len and not bool(seen[idx]):
                kept.append(idx)
                seen[idx] = True

        missing = torch.nonzero(~seen, as_tuple=False).flatten().tolist()

        return torch.tensor(kept + missing, device=device, dtype=torch.long)

    @torch.no_grad()
    def _sort_indices_by_metric(self, idx, scores, descending=True):
        if idx.numel() <= 1:
            return idx

        local_scores = scores[idx]
        order = torch.argsort(local_scores, descending=descending)

        return idx[order]

    @torch.no_grad()
    def _make_joint_order(self, tab_idx, img_idx, tab_scores, img_scores, O):
        mode = self.within_cluster_order

        if mode == "original":
            return torch.cat([tab_idx, O + img_idx])

        if mode == "metric_asc":
            tab_sorted = self._sort_indices_by_metric(
                tab_idx,
                tab_scores,
                descending=False,
            )
            img_sorted = self._sort_indices_by_metric(
                img_idx,
                img_scores,
                descending=False,
            )
            return torch.cat([tab_sorted, O + img_sorted])

        if mode == "metric_desc":
            tab_sorted = self._sort_indices_by_metric(
                tab_idx,
                tab_scores,
                descending=True,
            )
            img_sorted = self._sort_indices_by_metric(
                img_idx,
                img_scores,
                descending=True,
            )
            return torch.cat([tab_sorted, O + img_sorted])

        if mode == "alternating":
            tab_sorted = self._sort_indices_by_metric(
                tab_idx,
                tab_scores,
                descending=True,
            )
            img_sorted = self._sort_indices_by_metric(
                img_idx,
                img_scores,
                descending=True,
            )

            pieces = []

            for t, im in zip_longest(tab_sorted.tolist(), img_sorted.tolist()):
                if t is not None:
                    pieces.append(t)
                if im is not None:
                    pieces.append(O + im)

            return torch.tensor(pieces, device=tab_idx.device, dtype=torch.long)

        raise ValueError(
            f"Unknown within_cluster_order: {self.within_cluster_order}. "
            "Supported: metric_desc, metric_asc, original, alternating."
        )

    @torch.no_grad()
    def forward(self, tab_tokens, img_tokens):
        device = tab_tokens.device

        B, O, Ht = tab_tokens.shape
        B2, D, Hi = img_tokens.shape

        assert B == B2, "Tabular and image token batches must match."

        # Token identity vectors averaged over batch.
        tab_points = tab_tokens.mean(dim=0)  # (O,H)
        img_points = img_tokens.mean(dim=0)  # (D,H)

        lab_tab, _ = kmeans_torch(tab_points, k=self.k, device=device)
        lab_img, _ = kmeans_torch(img_points, k=self.k, device=device)

        kt = int(lab_tab.max().item()) + 1
        ki = int(lab_img.max().item()) + 1

        tab_idx = [
            torch.nonzero(lab_tab == i, as_tuple=False).flatten()
            for i in range(kt)
        ]
        img_idx = [
            torch.nonzero(lab_img == i, as_tuple=False).flatten()
            for i in range(ki)
        ]

        tab_scores = self._metric_values(tab_tokens)
        img_scores = self._metric_values(img_tokens)

        tab_energy, tab_centroid, tab_size = [], [], []
        img_energy, img_centroid, img_size = [], [], []

        for i in range(kt):
            e, c, s = self._cluster_energy_and_centroid(
                tab_tokens,
                tab_idx[i],
                tab_scores,
            )
            tab_energy.append(e)
            tab_centroid.append(c)
            tab_size.append(s)

        for i in range(ki):
            e, c, s = self._cluster_energy_and_centroid(
                img_tokens,
                img_idx[i],
                img_scores,
            )
            img_energy.append(e)
            img_centroid.append(c)
            img_size.append(s)

        Psi = torch.zeros((kt, ki), device=device)

        eps = 1e-8
        tau = max(self.sync_temperature, 1e-6)

        for r in range(kt):
            for c in range(ki):
                if tab_idx[r].numel() == 0 or img_idx[c].numel() == 0:
                    continue

                Et = tab_energy[r]
                Ei = img_energy[c]
                Ct = tab_centroid[r]
                Ci = img_centroid[c]

                if Et is None or Ei is None or Ct is None or Ci is None:
                    continue

                E = 1.0 - torch.abs(Et - Ei) / (Et + Ei + eps)
                E = E.clamp(0.0, 1.0)

                S = torch.dot(Ct, Ci).clamp(-1.0, 1.0)
                S = ((S + 1.0) / 2.0).clamp(0.0, 1.0)

                psi = (E.clamp_min(eps) ** self.energy_weight) * (
                    S.clamp_min(eps) ** self.centroid_weight
                )

                # Temperature: lower = sharper matching, higher = flatter matching.
                psi = psi ** (1.0 / tau)

                Psi[r, c] = psi

        # Fixed/default Hungarian matching.
        # This is not tuned. It is the NS-PFS cluster-pairing solver.
        rr, cc = linear_sum_assignment((-Psi).detach().cpu().numpy())

        pairs = []

        for r, c in zip(rr, cc):
            sync_score = float(Psi[r, c].detach().cpu().item())

            energy_score = float(
                (
                    tab_energy[r].detach().cpu()
                    + img_energy[c].detach().cpu()
                ).item()
            )

            size_score = int(tab_size[r] + img_size[c])
            pairs.append((r, c, sync_score, energy_score, size_score))

        # Tune/order only the matched joint clusters after Hungarian matching.
        if self.pair_order == "sync":
            pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
        elif self.pair_order == "energy":
            pairs = sorted(pairs, key=lambda x: x[3], reverse=True)
        elif self.pair_order == "size":
            pairs = sorted(pairs, key=lambda x: x[4], reverse=True)
        else:
            raise ValueError(
                f"Unknown pair_order: {self.pair_order}. "
                "Supported: sync, energy, size."
            )

        pieces = []

        for r, c, _, _, _ in pairs:
            joint = self._make_joint_order(
                tab_idx=tab_idx[r],
                img_idx=img_idx[c],
                tab_scores=tab_scores,
                img_scores=img_scores,
                O=O,
            )

            if joint.numel() > 0:
                pieces.append(joint)

        if len(pieces) > 0:
            seq = torch.cat(pieces).to(device).long()
        else:
            seq = torch.empty(0, device=device, dtype=torch.long)

        perm = self._order_preserving_unique_complete(
            seq=seq,
            total_len=O + D,
            device=device,
        )

        return perm

# ======================================================
# iSyncTab-AV Model
# ======================================================
class iSyncTab_AV(nn.Module):
    """Audio-video extension of iSyncTab using precomputed token features.

    Parameters
    ----------
    num_classes : int
        Number of target classes.
    audio_dim, video_dim : int
        Input feature dimensions of each audio/video token.
    audio_len, video_len : int
        Fixed number of tokens per modality. Linformer uses a fixed sequence
        length, so forward inputs must match these lengths.
    d_model : int
        Shared token embedding dimension.
    num_memory_tokens : int
        Learnable OMT-style memory tokens. Set to 0 to reproduce the tested
        AV variant that uses mean pooling over ordered data tokens; use 1 or
        more to enable the original memory-token readout.
    """

    def __init__(
        self,
        num_classes,
        audio_dim=64,
        video_dim=2048,
        audio_len=32,
        video_len=16,
        d_model=256,
        num_clusters=6,
        metric="variance",
        linformer_depth=4,
        linformer_heads=8,
        linformer_k=32,
        lambda_fs=0.28489162372638965,
        dropout=0.1354087338084182,
        num_memory_tokens=0,
        device=None,
        # Full NS-PFS tuning knobs retained from iSyncTab
        nspfs_bins=32,
        nspfs_mi_chunk=128,
        nspfs_sync_temperature=1.0,
        nspfs_energy_weight=1.0,
        nspfs_centroid_weight=1.0,
        nspfs_pair_order="sync",
        nspfs_within_cluster_order="metric_desc",
    ):
        super().__init__()

        self.lambda_fs = float(lambda_fs)
        self.audio_len = int(audio_len)
        self.video_len = int(video_len)
        self.num_memory_tokens = int(num_memory_tokens)
        self.data_seq_len = self.audio_len + self.video_len
        self.seq_len_max = self.num_memory_tokens + self.data_seq_len

        # Projection adapters only: feature extraction is performed upstream.
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.video_proj = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.audio_pos = nn.Parameter(
            torch.randn(1, self.audio_len, d_model) * 0.02
        )
        self.video_pos = nn.Parameter(
            torch.randn(1, self.video_len, d_model) * 0.02
        )

        self.nspfs = NSPFS_GPU(
            num_clusters=num_clusters,
            metric=metric,
            bins=nspfs_bins,
            mi_chunk=nspfs_mi_chunk,
            sync_temperature=nspfs_sync_temperature,
            energy_weight=nspfs_energy_weight,
            centroid_weight=nspfs_centroid_weight,
            pair_order=nspfs_pair_order,
            within_cluster_order=nspfs_within_cluster_order,
            device=device,
        )

        if self.num_memory_tokens > 0:
            self.mem_tokens = nn.Parameter(
                torch.randn(1, self.num_memory_tokens, d_model) * 0.02
            )
        else:
            self.mem_tokens = None

        self.linformer = Linformer(
            dim=d_model,
            seq_len=self.seq_len_max,
            depth=linformer_depth,
            heads=linformer_heads,
            k=linformer_k,
            one_kv_head=True,
            share_kv=True,
        )

        self.cls_head = nn.Linear(d_model, num_classes)
        self.seq_head = nn.Linear(d_model, 1)

    @staticmethod
    def _sanitize_perm(perm, length, device):
        """Keep a valid order-preserving permutation and append missing IDs."""
        perm = perm.detach().to(device).long().flatten()

        seen = torch.zeros(length, dtype=torch.bool, device=device)
        kept = []

        for idx in perm.tolist():
            if 0 <= idx < length and not bool(seen[idx]):
                kept.append(idx)
                seen[idx] = True

        missing = torch.nonzero(~seen, as_tuple=False).flatten().tolist()
        return torch.tensor(kept + missing, device=device, dtype=torch.long)

    def forward(self, x_audio, x_video, y=None):
        """Forward pass.

        Parameters
        ----------
        x_audio : Tensor [B, audio_len, audio_dim]
        x_video : Tensor [B, video_len, video_dim]
        y : optional LongTensor [B]

        Returns
        -------
        dict containing logits, NS-PFS permutation, sequence scores/targets,
        latent representations, and losses when labels are supplied.
        """
        if x_audio.ndim != 3 or x_video.ndim != 3:
            raise ValueError(
                "x_audio and x_video must both be rank-3 tensors: "
                "[batch, tokens, features]."
            )

        if x_audio.size(1) != self.audio_len:
            raise ValueError(
                f"Expected {self.audio_len} audio tokens, got {x_audio.size(1)}."
            )
        if x_video.size(1) != self.video_len:
            raise ValueError(
                f"Expected {self.video_len} video tokens, got {x_video.size(1)}."
            )

        audio_tok = self.audio_proj(x_audio) + self.audio_pos
        video_tok = self.video_proj(x_video) + self.video_pos

        B = audio_tok.size(0)
        L = self.data_seq_len

        # NS-PFS treats the two token streams exactly as the two modalities.
        perm = self.nspfs(audio_tok, video_tok)
        perm = self._sanitize_perm(perm, length=L, device=audio_tok.device)

        z_mm = torch.cat([audio_tok, video_tok], dim=1)
        z_pi = z_mm[:, perm, :]

        if self.num_memory_tokens > 0:
            mem = self.mem_tokens.expand(B, -1, -1)
            omt_input = torch.cat([mem, z_pi], dim=1)
        else:
            omt_input = z_pi

        if omt_input.size(1) != self.seq_len_max:
            raise RuntimeError(
                f"Linformer expected seq_len={self.seq_len_max}, "
                f"but received {omt_input.size(1)}."
            )

        h_all = self.linformer(omt_input)

        if self.num_memory_tokens > 0:
            h_mem = h_all[:, : self.num_memory_tokens, :]
            h_pi = h_all[:, self.num_memory_tokens :, :]
            h_cls = h_mem.mean(dim=1)
        else:
            h_mem = None
            h_pi = h_all
            h_cls = h_pi.mean(dim=1)

        logits = self.cls_head(h_cls)
        seq_scores = self.seq_head(h_pi).squeeze(-1)

        if L > 1:
            beta = torch.linspace(
                0.0,
                1.0,
                steps=L,
                device=logits.device,
                dtype=seq_scores.dtype,
            ).unsqueeze(0).expand(B, -1)
        else:
            beta = torch.zeros(
                B,
                L,
                device=logits.device,
                dtype=seq_scores.dtype,
            )

        out = {
            "logits": logits,
            "perm": perm,
            "seq_scores": seq_scores,
            "beta": beta,
            "h_cls": h_cls,
            "h_pi": h_pi,
        }

        if h_mem is not None:
            out["h_mem"] = h_mem

        if y is not None:
            ce = F.cross_entropy(logits, y)
            fs = F.mse_loss(seq_scores, beta)
            out["loss"] = ce + self.lambda_fs * fs
            out["loss_ce"] = ce
            out["loss_fs"] = fs

        return out


# Backward-friendly alias used in the experimental notebooks/scripts.
iSyncTabAV = iSyncTab_AV


if __name__ == "__main__":
    # Minimal shape smoke test with the RAVDESS feature convention.
    set_seed(42, deterministic=False)

    model = iSyncTab_AV(
        num_classes=8,
        audio_dim=64,
        video_dim=2048,
        audio_len=32,
        video_len=16,
        num_memory_tokens=0,
    )

    audio = torch.randn(2, 32, 64)
    video = torch.randn(2, 16, 2048)
    labels = torch.randint(0, 8, (2,))

    out = model(audio, video, labels)
    print("logits:", tuple(out["logits"].shape))
    print("perm:", tuple(out["perm"].shape))
    print("loss:", float(out["loss"].detach()))
