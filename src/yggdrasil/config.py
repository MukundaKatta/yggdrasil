"""
Configuration management for Yggdrasil.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Config:
    """Global configuration for a Yggdrasil instance.

    Attributes:
        default_dimension: Default vector dimensionality.
        default_metric: Default distance metric for search.
        decay_rate: Default decay rate for memory importance scoring.
        max_results: Default maximum search results.
    """

    default_dimension: int = 128
    default_metric: str = "cosine"
    decay_rate: float = 0.01
    max_results: int = 10

    @classmethod
    def from_env(cls) -> "Config":
        """Create a Config from environment variables.

        Environment variables:
            YGGDRASIL_DIMENSION: Default vector dimension.
            YGGDRASIL_METRIC: Default distance metric.
            YGGDRASIL_DECAY_RATE: Memory decay rate.
            YGGDRASIL_MAX_RESULTS: Default max results.
        """
        return cls(
            default_dimension=int(
                os.environ.get("YGGDRASIL_DIMENSION", "128")
            ),
            default_metric=os.environ.get("YGGDRASIL_METRIC", "cosine"),
            decay_rate=float(
                os.environ.get("YGGDRASIL_DECAY_RATE", "0.01")
            ),
            max_results=int(
                os.environ.get("YGGDRASIL_MAX_RESULTS", "10")
            ),
        )

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.default_dimension <= 0:
            raise ValueError("default_dimension must be positive")
        if self.default_metric not in ("cosine", "euclidean", "dot"):
            raise ValueError(
                f"Unknown metric '{self.default_metric}'. "
                "Choose from: cosine, euclidean, dot"
            )
        if self.decay_rate < 0:
            raise ValueError("decay_rate must be non-negative")
        if self.max_results <= 0:
            raise ValueError("max_results must be positive")
