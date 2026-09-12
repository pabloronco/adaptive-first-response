from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .backbone import GNNActorCritic
from .round_policy import RoundPolicy

# Engineering hardening (see docs/CHECKPOINT_ENGINEERING_REPORT.md follow-up):
# a checkpoint must be self-describing. Saving only a state_dict makes correct
# loading depend on the caller separately remembering/reconstructing the exact
# architecture (hidden_dim, num_layers, effort_per_pick) it was trained with -
# a silent source of load-time mismatches. Every checkpoint written here embeds
# its own architecture config, so load_policy_checkpoint() alone is enough to
# reconstruct an identical, ready-to-use policy.

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PolicyArchitectureConfig:
    hidden_dim: int
    num_layers: int
    effort_per_pick: int
    max_picks_per_round: int | None = None

    def build(self) -> RoundPolicy:
        backbone = GNNActorCritic(hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        return RoundPolicy(
            backbone,
            hidden_dim=self.hidden_dim,
            effort_per_pick=self.effort_per_pick,
            max_picks_per_round=self.max_picks_per_round,
        )


@dataclass(frozen=True)
class LoadedCheckpoint:
    policy: RoundPolicy
    architecture: PolicyArchitectureConfig
    update_idx: int
    extra: dict[str, Any]


def save_policy_checkpoint(
    path: str | Path,
    policy: RoundPolicy,
    *,
    architecture: PolicyArchitectureConfig,
    update_idx: int,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a single self-describing checkpoint file.

    Deliberately does not require an optimizer: inference-only deployment
    (e.g. demo mode) should not need to carry optimizer state around.
    """

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": asdict(architecture),
        "update_idx": update_idx,
        "policy_state_dict": policy.state_dict(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_policy_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> LoadedCheckpoint:
    """Reconstruct a ready-to-use `RoundPolicy` from a self-describing checkpoint.

    Raises `ValueError` on a checkpoint written by an incompatible/unknown
    format rather than silently misloading it.
    """

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "format_version" not in payload:
        raise ValueError(
            f"{path} does not look like a checkpoint written by "
            "save_policy_checkpoint() (missing format_version)."
        )
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format_version {payload['format_version']} is not "
            f"supported (expected {CHECKPOINT_FORMAT_VERSION})."
        )

    architecture = PolicyArchitectureConfig(**payload["architecture"])
    policy = architecture.build()
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()

    return LoadedCheckpoint(
        policy=policy,
        architecture=architecture,
        update_idx=payload["update_idx"],
        extra=payload.get("extra", {}),
    )


def load_optimizer_state(
    path: str | Path, optimizer: torch.optim.Optimizer, *, map_location: str | torch.device = "cpu"
) -> None:
    """Restore optimizer state from a checkpoint written with one, for resuming
    training (as opposed to inference-only loading via `load_policy_checkpoint`).
    """

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "optimizer_state_dict" not in payload:
        raise ValueError(f"{path} was saved without optimizer state.")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
