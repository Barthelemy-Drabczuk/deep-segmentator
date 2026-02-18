"""Dataset loader implementations."""
from .abcd_loader import ABCDLoader
from .abide_loader import ABIDELoader
from .custom_loader import CustomLoader
from .senior_loader import SENIORLoader
from .ukbiobank_loader import UKBiobankLoader

__all__ = [
    "UKBiobankLoader",
    "ABCDLoader",
    "ABIDELoader",
    "SENIORLoader",
    "CustomLoader",
]
