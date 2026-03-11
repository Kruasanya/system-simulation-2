from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SATISFIED_SHARE = 0.5
NEUTRAL_SHARE = 0.3
DISSATISFIED_SHARE = 0.2


@dataclass
class ModifiedBassModel:
    N: float = 10_000
    pA: float = 0.01
    qA: float = 0.30
    pB: float = 0.01
    qB: float = 0.30
    disappointment: float = 0.02
    tolerance: float = 0.05
    aggression: float = 0.03
    initial_A: float = 0.0
    initial_B: float = 0.0

    def __post_init__(self) -> None:
        if not np.isclose(
            SATISFIED_SHARE + NEUTRAL_SHARE + DISSATISFIED_SHARE, 1.0
        ):
            raise ValueError("Customer type shares must sum to 1.")
        if self.N <= 0:
            raise ValueError("Total market size N must be positive.")
        self.reset()

    def reset(self) -> None:
        self.potential = float(self.N - self.initial_A - self.initial_B)
        if self.potential < 0:
            raise ValueError("Initial customers exceed total market size.")

        self.A_satisfied = self.initial_A * SATISFIED_SHARE
        self.A_neutral = self.initial_A * NEUTRAL_SHARE
        self.A_dissatisfied = self.initial_A * DISSATISFIED_SHARE
        self.B_satisfied = self.initial_B * SATISFIED_SHARE
        self.B_neutral = self.initial_B * NEUTRAL_SHARE
        self.B_dissatisfied = self.initial_B * DISSATISFIED_SHARE

    @property
    def A_total(self) -> float:
        return self.A_satisfied + self.A_neutral + self.A_dissatisfied

    @property
    def B_total(self) -> float:
        return self.B_satisfied + self.B_neutral + self.B_dissatisfied

    @property
    def market_total(self) -> float:
        return self.potential + self.A_total + self.B_total

    @staticmethod
    def _bounded_flow(flow: float, available: float) -> float:
        return float(np.clip(flow, 0.0, max(available, 0.0)))

    def _customer_cost(self, satisfied: float, neutral: float, dissatisfied: float) -> float:
        hundred_blocks = np.ceil(max(satisfied, 0.0) / 100.0)
        capital_cost = 100.0 * hundred_blocks
        operating_cost = 5.0 * max(satisfied, 0.0)
        neutral_cost = 1.0 * max(neutral, 0.0)
        dissatisfied_cost = 4.0 * max(dissatisfied, 0.0)
        return capital_cost + operating_cost + neutral_cost + dissatisfied_cost

    def _record(self, step: int) -> dict[str, float]:
        adopted_total = self.A_total + self.B_total
        share_A = self.A_total / adopted_total if adopted_total > 0 else 0.0
        share_B = self.B_total / adopted_total if adopted_total > 0 else 0.0

        return {
            "step": step,
            "potential": self.potential,
            "A_satisfied": self.A_satisfied,
            "A_neutral": self.A_neutral,
            "A_dissatisfied": self.A_dissatisfied,
            "A_total": self.A_total,
            "B_satisfied": self.B_satisfied,
            "B_neutral": self.B_neutral,
            "B_dissatisfied": self.B_dissatisfied,
            "B_total": self.B_total,
            "share_A": share_A,
            "share_B": share_B,
            "cost_A": self._customer_cost(
                self.A_satisfied, self.A_neutral, self.A_dissatisfied
            ),
            "cost_B": self._customer_cost(
                self.B_satisfied, self.B_neutral, self.B_dissatisfied
            ),
        }

    def step(self) -> dict[str, float]:
        n = self.market_total
        vulnerable_A = self.A_neutral + self.A_dissatisfied
        vulnerable_B = self.B_neutral + self.B_dissatisfied

        adopt_A_raw = self.pA * self.potential + self.qA * self.A_satisfied * self.potential / n
        adopt_B_raw = self.pB * self.potential + self.qB * self.B_satisfied * self.potential / n

        switch_to_A_raw = self.aggression * self.tolerance * self.A_satisfied * vulnerable_B / n
        switch_to_B_raw = self.aggression * self.tolerance * self.B_satisfied * vulnerable_A / n

        return_A_raw = self.disappointment * self.A_dissatisfied
        return_B_raw = self.disappointment * self.B_dissatisfied

        adopt_A = self._bounded_flow(adopt_A_raw, self.potential)
        adopt_B = self._bounded_flow(adopt_B_raw, self.potential - adopt_A)

        switch_to_A = self._bounded_flow(switch_to_A_raw, vulnerable_B)
        switch_to_B = self._bounded_flow(switch_to_B_raw, vulnerable_A)

        return_A = self._bounded_flow(return_A_raw, self.A_dissatisfied)
        return_B = self._bounded_flow(return_B_raw, self.B_dissatisfied)

        self.potential -= adopt_A + adopt_B

        self.A_satisfied += adopt_A * SATISFIED_SHARE
        self.A_neutral += adopt_A * NEUTRAL_SHARE
        self.A_dissatisfied += adopt_A * DISSATISFIED_SHARE

        self.B_satisfied += adopt_B * SATISFIED_SHARE
        self.B_neutral += adopt_B * NEUTRAL_SHARE
        self.B_dissatisfied += adopt_B * DISSATISFIED_SHARE

        self.A_dissatisfied -= return_A
        self.B_dissatisfied -= return_B
        self.potential += return_A + return_B

        if vulnerable_B > 0:
            moved_B_neutral = switch_to_A * self.B_neutral / vulnerable_B
            moved_B_dissatisfied = switch_to_A * self.B_dissatisfied / vulnerable_B
        else:
            moved_B_neutral = 0.0
            moved_B_dissatisfied = 0.0

        if vulnerable_A > 0:
            moved_A_neutral = switch_to_B * self.A_neutral / vulnerable_A
            moved_A_dissatisfied = switch_to_B * self.A_dissatisfied / vulnerable_A
        else:
            moved_A_neutral = 0.0
            moved_A_dissatisfied = 0.0

        self.B_neutral -= moved_B_neutral
        self.B_dissatisfied -= moved_B_dissatisfied
        self.A_neutral -= moved_A_neutral
        self.A_dissatisfied -= moved_A_dissatisfied

        self.A_neutral += moved_B_neutral
        self.A_dissatisfied += moved_B_dissatisfied
        self.B_neutral += moved_A_neutral
        self.B_dissatisfied += moved_A_dissatisfied

        for attr in (
            "potential",
            "A_satisfied",
            "A_neutral",
            "A_dissatisfied",
            "B_satisfied",
            "B_neutral",
            "B_dissatisfied",
        ):
            setattr(self, attr, max(getattr(self, attr), 0.0))

        return self._record(-1)

    def run(self, steps: int = 200, reset: bool = True) -> pd.DataFrame:
        if reset:
            self.reset()

        history = [self._record(0)]
        for step in range(1, steps + 1):
            self.step()
            history.append(self._record(step))
        return pd.DataFrame(history)

    def convergence_summary(
        self, steps: int = 200, window: int = 20, tolerance_value: float = 1e-3
    ) -> dict[str, float | bool]:
        history = self.run(steps=steps, reset=True)
        tail = history["share_A"].tail(window)
        max_deviation = float((tail - tail.mean()).abs().max())
        return {
            "equilibrium_share_A": float(history["share_A"].iloc[-1]),
            "equilibrium_share_B": float(history["share_B"].iloc[-1]),
            "max_share_deviation": max_deviation,
            "converged": bool(max_deviation <= tolerance_value),
        }


def simulate_equilibrium_share(
    pA: float,
    pB: float,
    *,
    qA: float = 0.30,
    qB: float = 0.30,
    disappointment: float = 0.02,
    tolerance: float = 0.05,
    aggression: float = 0.03,
    steps: int = 200,
) -> float:
    model = ModifiedBassModel(
        pA=pA,
        qA=qA,
        pB=pB,
        qB=qB,
        disappointment=disappointment,
        tolerance=tolerance,
        aggression=aggression,
    )
    history = model.run(steps=steps)
    return float(history["share_A"].iloc[-1])


def generate_dataset(
    pA_values: np.ndarray,
    pB_values: np.ndarray,
    *,
    qA: float = 0.30,
    qB: float = 0.30,
    disappointment: float = 0.02,
    tolerance: float = 0.05,
    aggression: float = 0.03,
    steps: int = 200,
) -> pd.DataFrame:
    rows: list[dict[str, float | bool]] = []
    for pA in pA_values:
        for pB in pB_values:
            model = ModifiedBassModel(
                pA=float(pA),
                qA=qA,
                pB=float(pB),
                qB=qB,
                disappointment=disappointment,
                tolerance=tolerance,
                aggression=aggression,
            )
            summary = model.convergence_summary(steps=steps)
            rows.append(
                {
                    "pA": float(pA),
                    "pB": float(pB),
                    "share_A": float(summary["equilibrium_share_A"]),
                    "share_B": float(summary["equilibrium_share_B"]),
                    "converged": bool(summary["converged"]),
                    "max_share_deviation": float(summary["max_share_deviation"]),
                }
            )
    return pd.DataFrame(rows)
