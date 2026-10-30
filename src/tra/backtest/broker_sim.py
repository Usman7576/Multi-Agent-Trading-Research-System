"""Simulated broker: orders placed after a session's close fill at the next session's open.

Prices are total-return adjusted, so dividends accrue in the price series. Sells execute before buys;
buys are scaled down if cash (after costs) is insufficient. There is no leverage and no shorting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tra.backtest.costs import CostModel


@dataclass(frozen=True)
class Order:
    ticker: str
    shares: float  # signed: > 0 buy, < 0 sell
    created: pd.Timestamp


@dataclass(frozen=True)
class Fill:
    date: pd.Timestamp
    ticker: str
    shares: float
    price: float
    value: float
    cost: float
    order_created: pd.Timestamp


@dataclass(frozen=True)
class DayLimits:
    """Market frictions for one session: circuit-breaker locks and a cap on volume participation."""

    buy_blocked: frozenset[str] = frozenset()  # locked at the upper limit all day: no sellers
    sell_blocked: frozenset[str] = frozenset()  # locked at the lower limit all day: no buyers
    traded_value: pd.Series | None = None  # the day's traded value per ticker
    max_participation: float | None = None


@dataclass
class SimBroker:
    cash: float
    costs: CostModel
    positions: dict[str, float] = field(default_factory=dict)
    pending: list[Order] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    unfilled: list[tuple[pd.Timestamp, Order]] = field(default_factory=list)

    def submit(self, orders: list[Order]) -> None:
        self.pending.extend(o for o in orders if o.shares != 0)

    def fill_pending_orders(
        self,
        day: pd.Timestamp,
        open_px: pd.Series,
        adv: pd.Series,
        daily_vol: pd.Series,
        limits: DayLimits | None = None,
    ) -> list[Fill]:
        """Fill every pending order at ``day``'s open. Orders created on ``day`` itself are rejected.

        With ``limits``: an order into a stock locked at its circuit limit does not fill, and an order is cut
        to ``max_participation`` of the day's traded value. Unfilled parts are dropped, not carried over.
        """
        todays: list[Fill] = []
        orders, self.pending = self.pending, []
        for order in sorted(orders, key=lambda o: o.shares):  # sells (negative) first
            if order.created >= day:
                raise RuntimeError(f"Order created {order.created} cannot fill on the same bar {day}")
            price = open_px.get(order.ticker, np.nan)
            if not np.isfinite(price) or price <= 0:
                self.unfilled.append((day, order))
                continue
            shares = order.shares
            if shares < 0:
                shares = max(shares, -self.positions.get(order.ticker, 0.0))
            if limits is not None:
                blocked = limits.buy_blocked if shares > 0 else limits.sell_blocked
                if order.ticker in blocked:
                    self.unfilled.append((day, order))
                    continue
                if limits.max_participation is not None and limits.traded_value is not None:
                    cap = limits.max_participation * float(limits.traded_value.get(order.ticker, 0.0))
                    if abs(shares * price) > cap:
                        self.unfilled.append((day, order))
                        shares = np.sign(shares) * cap / price
            value = shares * price
            rate = self.costs.cost_rate(
                value, adv.get(order.ticker, np.nan), daily_vol.get(order.ticker, np.nan)
            )
            if shares > 0 and value * (1 + rate) > self.cash:
                shares = max(self.cash / (price * (1 + rate)), 0.0)
                value = shares * price
            if shares == 0:
                continue
            cost = abs(value) * rate
            self.cash -= value + cost
            self.positions[order.ticker] = self.positions.get(order.ticker, 0.0) + shares
            if abs(self.positions[order.ticker]) < 1e-9:
                del self.positions[order.ticker]
            todays.append(Fill(day, order.ticker, shares, price, value, cost, order.created))
        self.fills.extend(todays)
        return todays

    def nav(self, px: pd.Series) -> float:
        return self.cash + sum(sh * px.get(t, np.nan) for t, sh in self.positions.items())

    def weights(self, px: pd.Series) -> pd.Series:
        nav = self.nav(px)
        return pd.Series({t: sh * px.get(t, np.nan) / nav for t, sh in self.positions.items()}, dtype=float)

    def orders_for_target(
        self, target: pd.Series, px: pd.Series, created: pd.Timestamp, min_trade_weight: float
    ) -> list[Order]:
        """Convert target weights into share orders using ``created``'s close (fills at next open)."""
        nav = self.nav(px)
        current = self.weights(px)
        tickers = sorted(set(target.index) | set(current.index))
        orders = []
        for t in tickers:
            w_tgt = float(target.get(t, 0.0))
            w_cur = float(current.get(t, 0.0))
            price = px.get(t, np.nan)
            if not np.isfinite(price) or price <= 0:
                continue
            closing = w_tgt == 0.0 and w_cur != 0.0
            if abs(w_tgt - w_cur) < min_trade_weight and not closing:
                continue
            shares = (w_tgt - w_cur) * nav / price
            if closing:
                shares = -self.positions.get(t, 0.0)
            orders.append(Order(t, shares, created))
        return orders
