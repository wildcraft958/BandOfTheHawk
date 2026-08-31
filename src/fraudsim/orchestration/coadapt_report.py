"""Co-adaptation reporting and progress display."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..logs import get_logger
from ..protocols import RiskAction

_REFUSING_ACTIONS = frozenset(
    {RiskAction.HOLD, RiskAction.DECLINE, RiskAction.BLOCK}
)

_log = get_logger(__name__)


class _Progress:
    """Announces a long step and how long it took.

    The live phase reports per update so that a slow run cannot be mistaken for a
    hung one. These are diagnostics, so they go to the log on stderr, leaving the
    rendered report alone on stdout.
    """

    __slots__ = ("_label", "_started")

    def __init__(self, label: str) -> None:
        self._label = label
        self._started = time.perf_counter()
        _log.info("%s", label)

    def say(self, message: str) -> None:
        _log.info("  [%5.1fs] %s", time.perf_counter() - self._started, message)

    def done(self, note: str = "") -> None:
        elapsed = time.perf_counter() - self._started
        tail = f"  ({note})" if note else ""
        _log.info("  %s done in %.1fs%s", self._label, elapsed, tail)


@dataclass
class CoadaptReport:
    """The live-phase curve and the phase-boundary facts."""

    initial_defender_positives: int = 0
    bc_final_loss: float = 0.0
    critic_final_loss: float = 0.0
    attacker_success: list[float] = field(default_factory=list)
    mean_return: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    critic_relative: list[float] = field(default_factory=list)
    defender_refits: list[int] = field(default_factory=list)
    defender_positives_at_refit: list[int] = field(default_factory=list)
    zero_shot: dict[str, float] = field(default_factory=dict)
    false_positive_rate: list[float] = field(default_factory=list)
    top_sequences: list[tuple[str, int]] = field(default_factory=list)
    strategy_history: list[dict] = field(default_factory=list)
    selection: dict = field(default_factory=dict)
    checkpoints: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Everything as plain data, for plotting and for the writeup."""
        return {
            "initial_defender_positives": self.initial_defender_positives,
            "bc_final_loss": self.bc_final_loss,
            "critic_final_loss": self.critic_final_loss,
            "attacker_success": list(self.attacker_success),
            "mean_return": list(self.mean_return),
            "entropy": list(self.entropy),
            "critic_relative": list(self.critic_relative),
            "defender_refits": list(self.defender_refits),
            "defender_positives_at_refit": list(self.defender_positives_at_refit),
            "zero_shot": dict(self.zero_shot),
            "false_positive_rate": list(self.false_positive_rate),
            "top_sequences": [
                {"sequence": seq, "count": count} for seq, count in self.top_sequences
            ],
            "strategy_history": list(self.strategy_history),
            "selection": dict(self.selection),
            "checkpoints": dict(self.checkpoints),
        }

    def render(self) -> str:
        lines = [
            "live co-adaptation",
            f"  initial defender fraud   {self.initial_defender_positives:>8,}",
            f"  BC final loss            {self.bc_final_loss:>8.4f}",
            f"  critic final loss        {self.critic_final_loss:>8.4f}",
            "",
            "  live phase  (extracted = value the attacker takes per episode, all channels)",
            "  update  extracted   return   entropy   defender",
        ]
        refit_set = set(self.defender_refits)
        for i, (succ, ret, ent) in enumerate(
            zip(self.attacker_success, self.mean_return, self.entropy)
        ):
            marker = "  <- refit" if i in refit_set else ""
            lines.append(f"    {i:<7}{succ:>9.1f}{ret:>9.2f}{ent:>10.3f}{marker}")
        lines += ["", "  reads"]
        if self.attacker_success:
            lines.append(
                f"    value extracted per episode  {self.attacker_success[0]:.1f}"
                f" -> {self.attacker_success[-1]:.1f}"
            )
        lines += ["", "  zero-shot recall on held-out verticals"]
        for name, recall in self.zero_shot.items():
            lines.append(f"    {name:<16}{recall:>8.3f}")
        if self.strategy_history:
            lines += ["", "  how the attacker's strategy changed (sampled at each refit)"]
            for snap in self.strategy_history:
                top = snap["sequences"][0] if snap["sequences"] else None
                if top:
                    lines.append(
                        f"    update {snap['update']:<4} {top['count']:>3}x  "
                        f"{top['sequence'][:92]}"
                    )

        if self.selection.get("describe"):
            lines += ["", self.selection["describe"]]

        lines += ["", "  top action sequences (final trained attacker)"]
        for seq, count in self.top_sequences[:8]:
            lines.append(f"    {count:>4}  {seq}")
        return "\n".join(lines)
