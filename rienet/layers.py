"""
Convenience access to the main RIEnet layer.

Preferred location for learnable layers is `rienet.trainable_layers`.
"""

from .trainable_layers import RIEnetLayer

__all__ = [
    "RIEnetLayer",
]
