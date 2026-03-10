"""
liquidity_filter.py
Filters out trades during low liquidity conditions.

Low liquidity = higher slippage, easier manipulation, unrealistic backtest assumptions.
"""

import ccxt
import numpy as np
from typing import Dict, Optional
from .base_filter import BaseFilter


class LiquidityFilter(BaseFilter):
    """
    Veto signals when trading volume is too low.

    Low volume indicates:
    - High slippage on entry/exit
    - Possible price manipulation
    - Thin order books
    - Backtest assumptions invalid
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.min_volume_ratio = self.config.get('min_volume_ratio', 0.5)  # 50% of avg
        self.lookback_days = self.config.get('lookback_days', 30)
        self._exchange = None

    def _get_exchange(self):
        """Lazy initialize exchange connection."""
        if self._exchange is None:
            self._exchange = ccxt.binance({'enableRateLimit': True})
        return self._exchange

    def _get_volume_stats(self, symbol: str) -> Dict:
        """
        Get current and average volume for symbol.

        Args:
            symbol: Trading pair (e.g., 'LINK/USDT')

        Returns:
            Dict with 'current_volume_24h' and 'avg_volume_30d'
        """
        try:
            exchange = self._get_exchange()

            # Fetch daily candles for lookback period
            daily_candles = exchange.fetch_ohlcv(
                symbol,
                timeframe='1d',
                limit=self.lookback_days + 1
            )

            if len(daily_candles) < 5:
                # Not enough data - fail open
                return {
                    'current_volume_24h': 1e9,  # Large number to pass
                    'avg_volume_30d': 1e9
                }

            # Extract volumes (index 5 in OHLCV)
            volumes = np.array([candle[5] for candle in daily_candles])

            # Current 24h volume (most recent candle)
            current_volume = volumes[-1]

            # Average volume over lookback period
            avg_volume = np.mean(volumes[:-1])  # Exclude current day

            return {
                'current_volume_24h': current_volume,
                'avg_volume_30d': avg_volume
            }

        except Exception as e:
            print(f"  ⚠️  Volume calculation error for {symbol}: {e}")
            # Fail open - don't veto on error
            return {
                'current_volume_24h': 1e9,
                'avg_volume_30d': 1e9
            }

    def filter_signal(self, signal: Dict) -> Dict:
        """
        Check if trading volume is sufficient.

        Args:
            signal: Trading signal dict

        Returns:
            Filter result with approval decision
        """
        if not self.enabled:
            return {
                'approved': True,
                'confidence': signal['prob'],
                'reason': 'Liquidity filter disabled'
            }

        symbol = signal['symbol']
        volume_stats = self._get_volume_stats(symbol)

        current_vol = volume_stats['current_volume_24h']
        avg_vol = volume_stats['avg_volume_30d']

        # Avoid division by zero
        if avg_vol < 1:
            return {
                'approved': False,
                'confidence': 0.0,
                'reason': 'No historical volume data',
                'metadata': volume_stats
            }

        volume_ratio = current_vol / avg_vol

        if volume_ratio < self.min_volume_ratio:
            return {
                'approved': False,
                'confidence': 0.0,
                'reason': f'Low liquidity: volume {volume_ratio:.1%} of 30d average (min {self.min_volume_ratio:.1%})',
                'metadata': {
                    'current_volume_24h': current_vol,
                    'avg_volume_30d': avg_vol,
                    'volume_ratio': volume_ratio,
                    'min_required': self.min_volume_ratio
                }
            }

        return {
            'approved': True,
            'confidence': signal['prob'],
            'reason': f'Liquidity OK: volume {volume_ratio:.1%} of average',
            'metadata': {
                'volume_ratio': volume_ratio,
                'current_volume_24h': current_vol
            }
        }

    def get_name(self) -> str:
        return "Liquidity Filter"


if __name__ == "__main__":
    # Test the filter
    filter_obj = LiquidityFilter({'enabled': True, 'min_volume_ratio': 0.5})

    test_signal = {
        'symbol': 'LINK/USDT',
        'signal': 'SHORT',
        'prob': 0.67,
        'strategy': 'mean_reversion'
    }

    result = filter_obj.filter_signal(test_signal)
    print(f"Filter: {filter_obj.get_name()}")
    print(f"Result: {result}")