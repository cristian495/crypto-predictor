"""
filters package
Signal filtering system for trading strategies.

Usage:
    from filters import FilterPipeline

    pipeline = FilterPipeline(config)
    filtered_signals = pipeline.filter_signals(signals)
"""

from typing import List, Dict, Optional
from .base_filter import BaseFilter
from .volatility_filter import VolatilityFilter
from .liquidity_filter import LiquidityFilter


class FilterPipeline:
    """
    Manages multiple filters in a pipeline.

    Filters are applied in sequence. If any filter vetos a signal,
    processing stops and the signal is rejected.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize filter pipeline from configuration.

        Args:
            config: Dictionary with filter configurations:
                {
                    'filters': {
                        'volatility': {'enabled': True, 'max_volatility': 0.10},
                        'liquidity': {'enabled': True, 'min_volume_ratio': 0.5}
                    }
                }
        """
        self.config = config or {}
        self.filters: List[BaseFilter] = []
        self._initialize_filters()

    def _initialize_filters(self):
        """Build filter instances from config."""
        filter_configs = self.config.get('filters', {})

        # Add filters in priority order (most critical first)

        # 1. Liquidity (critical - can't trade without it)
        if filter_configs.get('liquidity', {}).get('enabled', False):
            self.filters.append(LiquidityFilter(filter_configs['liquidity']))

        # 2. Volatility (protects from extreme conditions)
        if filter_configs.get('volatility', {}).get('enabled', False):
            self.filters.append(VolatilityFilter(filter_configs['volatility']))

        # Future filters can be added here
        # if filter_configs.get('btc_correlation', {}).get('enabled', False):
        #     self.filters.append(BTCCorrelationFilter(...))

        if self.filters:
            print(f"✅ Loaded {len(self.filters)} filter(s):")
            for f in self.filters:
                print(f"   - {f}")
        else:
            print(f"⚠️  No filters enabled")

    def filter_signals(self, signals: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """
        Run all signals through the filter pipeline.

        Args:
            signals: List of signal dictionaries from strategies

        Returns:
            Tuple of (approved_signals, rejected_signals)
        """
        if not self.filters:
            # No filters enabled - return all signals unchanged
            return signals, []

        filtered_signals = []
        rejected_signals = []

        for signal in signals:
            result = self._filter_single_signal(signal)

            if result['approved']:
                # Add filter metadata to signal
                signal['original_prob'] = signal.get('original_prob', signal['prob'])
                signal['prob'] = result['confidence']
                signal['filter_metadata'] = result.get('metadata', {})
                signal['filter_reasons'] = result.get('reasons', [])
                filtered_signals.append(signal)
            else:
                # Log rejection and save it
                print(f"  ❌ Filtered out {signal['symbol']} {signal['signal']}: "
                      f"{result['reason']}")
                # Add rejection reason to signal
                signal['rejection_reason'] = result['reason']
                signal['rejection_metadata'] = result.get('metadata', {})
                rejected_signals.append(signal)

        return filtered_signals, rejected_signals

    def _filter_single_signal(self, signal: Dict) -> Dict:
        """
        Run a single signal through all filters.

        Returns filter result with approval status.
        """
        current_confidence = signal['prob']
        reasons = []
        metadata = {}

        for filter_obj in self.filters:
            # Create a temporary signal dict with current confidence
            temp_signal = signal.copy()
            temp_signal['prob'] = current_confidence

            result = filter_obj.filter_signal(temp_signal)

            if not result['approved']:
                # Veto - stop processing immediately
                return {
                    'approved': False,
                    'confidence': 0.0,
                    'reason': f"{filter_obj.get_name()}: {result['reason']}",
                    'metadata': result.get('metadata', {})
                }

            # Update confidence (filter may have adjusted it)
            current_confidence = result.get('confidence', current_confidence)
            reasons.append(f"{filter_obj.get_name()}: {result.get('reason', 'OK')}")
            metadata[filter_obj.get_name()] = result.get('metadata', {})

        return {
            'approved': True,
            'confidence': current_confidence,
            'reasons': reasons,
            'metadata': metadata
        }

    def get_filter_summary(self) -> str:
        """Get a summary of active filters."""
        if not self.filters:
            return "No filters active"

        summary = f"{len(self.filters)} filter(s) active:\n"
        for f in self.filters:
            summary += f"  - {f}\n"
        return summary.strip()


# Convenience exports
__all__ = [
    'FilterPipeline',
    'BaseFilter',
    'VolatilityFilter',
    'LiquidityFilter'
]