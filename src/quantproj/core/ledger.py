"""Self-financing ledger in a deterministic or stochastic bank account."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Ledger:
    cash: float
    bank: float
    holdings: np.ndarray
    cost_pv: float = 0.0

    def accrue(self, bank: float) -> None:
        if bank <= 0:
            raise ValueError("Bank account must be positive.")
        self.cash *= bank / self.bank
        self.bank = bank

    def settle(self, cashflows) -> None:
        self.cash += float(self.holdings @ cashflows)

    def trade(self, target, marks, spreads) -> None:
        trades = target - self.holdings
        costs = float(abs(trades) @ spreads)
        self.cash -= float(trades @ marks) + costs
        self.holdings = np.array(target, copy=True)
        self.cost_pv += costs / self.bank

    def discounted(self, marks=None) -> float:
        marked = 0.0 if marks is None else self.holdings @ marks
        return float((self.cash + marked) / self.bank)
