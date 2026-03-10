"""
base_filter.py
Abstract base class for all signal filters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseFilter(ABC):
    """
    Base class for all signal filters.

    A filter takes a trading signal and either:
    1. Approves it (possibly with adjusted confidence)
    2. Rejects it (veto)

    Filters should be fast, focused, and fail-open (approve if error).
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize filter with configuration.

        Args:
            config: Dictionary with filter-specific settings.
                   Must include 'enabled' key.
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)

    @abstractmethod
    def filter_signal(self, signal: Dict) -> Dict:
        """
        Filter/validate a trading signal.

        Args:
            signal: Dictionary with signal information:
                {
                    'symbol': 'ETH/USDT',
                    'signal': 'LONG' or 'SHORT',
                    'prob': 0.67,
                    'strategy': 'mean_reversion',
                    'extra': 'Z=2.12 agree=3/3',
                    'timestamp': datetime (optional)
                }

        Returns:
            Dictionary with filter decision:
                {
                    'approved': True/False,
                    'confidence': 0.0-1.0,  # Original or adjusted
                    'reason': 'explanation string',
                    'metadata': {...}  # Optional additional info
                }

        Important:
            - If approved=False, the signal will be vetoed
            - confidence can be adjusted (e.g., 0.67 -> 0.55)
            - Always provide a clear 'reason' for logging
            - Fail open: if filter errors, return approved=True
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return filter name for logging."""
        pass

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"{self.get_name()} ({status})"
