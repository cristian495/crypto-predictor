"""
volatility_filter.py
Filters out trades during extreme volatility conditions.

Protects against trading when market conditions are outside
the model's training distribution.
"""

import ccxt
import numpy as np
from typing import Dict, Optional
from .base_filter import BaseFilter


class VolatilityFilter(BaseFilter):
    """
    Veto signals when realized volatility is too high.

    High volatility = unpredictable price action, wider slippage,
    SL can be jumped, model predictions less reliable.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.max_volatility = self.config.get('max_volatility', 0.10)  # 10% default
        self.lookback_hours = self.config.get('lookback_hours', 24)
        self._exchange = None

    def _get_exchange(self):
        """Lazy initialize exchange connection."""
        if self._exchange is None:
            self._exchange = ccxt.binance({'enableRateLimit': True})
        return self._exchange

    def _calculate_realized_volatility(self, symbol: str) -> float:
        """
        Calculate realized volatility over lookback period.

        Uses standard deviation of log returns (annualized equivalent).

        Args:
            symbol: Trading pair (e.g., 'LINK/USDT')

        Returns:
            Volatility as decimal (e.g., 0.05 = 5%)
        """
        try:
            exchange = self._get_exchange()

            # Fetch recent candles
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe='1h',
                limit=self.lookback_hours + 1
            )

            if len(ohlcv) < 10:
                # Not enough data - fail open
                return 0.0

            # Extract closing prices
            closes = np.array([candle[4] for candle in ohlcv])

            # Calculate log returns
            log_returns = np.log(closes[1:] / closes[:-1])

            # Realized volatility (std of returns)
            volatility = np.std(log_returns)

            # Annualize (sqrt of periods per year)
            # For 1h candles: sqrt(24 * 365) ≈ 93
            # But we want comparable to daily %, so use sqrt(24) ≈ 5
            volatility_normalized = volatility * np.sqrt(24)

            return volatility_normalized

        except Exception as e:
            print(f"  ⚠️  Volatility calculation error for {symbol}: {e}")
            # Fail open - don't veto on error
            return 0.0

    def filter_signal(self, signal: Dict) -> Dict:
        """
        Check if volatility is within acceptable range.

        Args:
            signal: Trading signal dict

        Returns:
            Filter result with approval decision
        """
        if not self.enabled:
            return {
                'approved': True,
                'confidence': signal['prob'],
                'reason': 'Volatility filter disabled'
            }

        symbol = signal['symbol']
        current_vol = self._calculate_realized_volatility(symbol)

        if current_vol > self.max_volatility:
            return {
                'approved': False,
                'confidence': 0.0,
                'reason': f'Volatility too high: {current_vol:.2%} > {self.max_volatility:.2%}',
                'metadata': {
                    'realized_volatility': current_vol,
                    'max_allowed': self.max_volatility
                }
            }

        return {
            'approved': True,
            'confidence': signal['prob'],
            'reason': f'Volatility OK: {current_vol:.2%}',
            'metadata': {
                'realized_volatility': current_vol
            }
        }

    def get_name(self) -> str:
        return "Volatility Filter"


if __name__ == "__main__":
    # Test the filter
    filter_obj = VolatilityFilter({'enabled': True, 'max_volatility': 0.10})

    test_signal = {
        'symbol': 'LINK/USDT',
        'signal': 'SHORT',
        'prob': 0.67,
        'strategy': 'mean_reversion'
    }

    result = filter_obj.filter_signal(test_signal)
    print(f"Filter: {filter_obj.get_name()}")
    print(f"Result: {result}")