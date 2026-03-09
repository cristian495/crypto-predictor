"""
metrics.py
Performance evaluation metrics for trading strategies.
Replaces AUC-focused metrics with trading-relevant ones.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(equity_curve: np.ndarray,
                 periods_per_year: int = 8760) -> float:
    """
    Annualized Sharpe ratio from equity curve.
    periods_per_year: 8760 for 1H candles (24 * 365).
    """
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def sortino_ratio(equity_curve: np.ndarray,
                  periods_per_year: int = 8760) -> float:
    """
    Annualized Sortino ratio (penalizes only downside volatility).
    """
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return float("inf") if np.mean(returns) > 0 else 0.0
    return float(np.mean(returns) / np.std(downside) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as a negative fraction (e.g., -0.10 = -10%)."""
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(dd.min())


def expectancy(trades_df: pd.DataFrame,
               pnl_col: str = "pnl_after_fees") -> float:
    """
    Expected value per trade.
    E = (win_rate * avg_win) - (loss_rate * avg_loss)
    """
    if len(trades_df) == 0:
        return 0.0

    wins = trades_df[trades_df[pnl_col] > 0][pnl_col]
    losses = trades_df[trades_df[pnl_col] <= 0][pnl_col]

    win_rate = len(wins) / len(trades_df)
    loss_rate = len(losses) / len(trades_df)

    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0

    return float(win_rate * avg_win - loss_rate * avg_loss)


def profit_factor(trades_df: pd.DataFrame,
                  pnl_col: str = "pnl_after_fees") -> float:
    """Gross profit / gross loss. > 1 means profitable."""
    if len(trades_df) == 0:
        return 0.0

    gross_profit = trades_df.loc[trades_df[pnl_col] > 0, pnl_col].sum()
    gross_loss = abs(trades_df.loc[trades_df[pnl_col] < 0, pnl_col].sum())

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def calmar_ratio(equity_curve: np.ndarray,
                 periods_per_year: int = 8760) -> float:
    """Annualized return / max drawdown."""
    if len(equity_curve) < 2:
        return 0.0

    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    n_periods = len(equity_curve)
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return float("inf") if annual_return > 0 else 0.0
    return float(annual_return / mdd)


def compute_all_metrics(
    trades_df: pd.DataFrame,
    equity_curve: np.ndarray,
    initial_capital: float,
    periods_per_year: int = 8760,
    pnl_col: str = "pnl_after_fees",
) -> dict:
    """
    Compute all performance metrics.

    Returns dict with: sharpe, sortino, expectancy, profit_factor,
    max_drawdown, calmar, total_return, win_rate, total_trades,
    avg_trade_duration, avg_win, avg_loss, best_trade, worst_trade.
    """
    total_return = (
        (equity_curve[-1] - initial_capital) / initial_capital
        if len(equity_curve) > 0
        else 0.0
    )

    # Win/loss stats
    if len(trades_df) > 0:
        wins = trades_df[trades_df[pnl_col] > 0]
        losses = trades_df[trades_df[pnl_col] <= 0]
        win_rate = len(wins) / len(trades_df)
        avg_win = wins[pnl_col].mean() if len(wins) > 0 else 0.0
        avg_loss = losses[pnl_col].mean() if len(losses) > 0 else 0.0
        best_trade = trades_df[pnl_col].max()
        worst_trade = trades_df[pnl_col].min()
    else:
        win_rate = avg_win = avg_loss = best_trade = worst_trade = 0.0

    # Average trade duration
    avg_duration = 0.0
    if len(trades_df) > 0 and "duration_bars" in trades_df.columns:
        avg_duration = trades_df["duration_bars"].mean()

    # Long/short breakdown
    n_long = n_short = 0
    if len(trades_df) > 0 and "direction" in trades_df.columns:
        n_long = (trades_df["direction"] == 1).sum()
        n_short = (trades_df["direction"] == -1).sum()

    return {
        "total_return": round(total_return, 4),
        "sharpe_ratio": round(sharpe_ratio(equity_curve, periods_per_year), 2),
        "sortino_ratio": round(sortino_ratio(equity_curve, periods_per_year), 2),
        "max_drawdown": round(max_drawdown(equity_curve), 4),
        "calmar_ratio": round(calmar_ratio(equity_curve, periods_per_year), 2),
        "expectancy": round(expectancy(trades_df, pnl_col), 6),
        "profit_factor": round(profit_factor(trades_df, pnl_col), 2),
        "win_rate": round(win_rate, 4),
        "total_trades": len(trades_df),
        "long_trades": int(n_long),
        "short_trades": int(n_short),
        "avg_trade_duration": round(avg_duration, 1),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "best_trade": round(best_trade, 4),
        "worst_trade": round(worst_trade, 4),
        "initial_capital": initial_capital,
        "final_capital": round(
            equity_curve[-1] if len(equity_curve) > 0 else initial_capital, 2
        ),
    }


def print_performance_report(metrics: dict, title: str = "PERFORMANCE REPORT"):
    """Pretty-print all performance metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    print(f"\n  --- Capital ---")
    print(f"  Initial:          ${metrics['initial_capital']:,.2f}")
    print(f"  Final:            ${metrics['final_capital']:,.2f}")
    print(f"  Total return:     {metrics['total_return']:.2%}")

    print(f"\n  --- Risk-Adjusted ---")
    print(f"  Sharpe ratio:     {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino ratio:    {metrics['sortino_ratio']:.2f}")
    print(f"  Calmar ratio:     {metrics['calmar_ratio']:.2f}")
    print(f"  Max drawdown:     {metrics['max_drawdown']:.2%}")

    print(f"\n  --- Trade Statistics ---")
    print(f"  Total trades:     {metrics['total_trades']}")
    print(f"  Long / Short:     {metrics['long_trades']} / {metrics['short_trades']}")
    print(f"  Win rate:         {metrics['win_rate']:.2%}")
    print(f"  Profit factor:    {metrics['profit_factor']:.2f}")
    print(f"  Expectancy:       {metrics['expectancy']:.4%} per trade")
    print(f"  Avg duration:     {metrics['avg_trade_duration']:.0f} bars")

    print(f"\n  --- PnL ---")
    print(f"  Avg win:          {metrics['avg_win']:.2%}")
    print(f"  Avg loss:         {metrics['avg_loss']:.2%}")
    print(f"  Best trade:       {metrics['best_trade']:.2%}")
    print(f"  Worst trade:      {metrics['worst_trade']:.2%}")

    print(f"{'=' * 60}")
