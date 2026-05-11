"""
Canonical Dictionary Builder.

Creates a clean, hand-curated canonical dictionary from the known merchant taxonomy.
This simulates a production system where canonical names are maintained
as a master reference.
"""

from typing import List

# Hand-curated canonical names based on the merchant taxonomy
CANONICAL_MERCHANTS = [
    "STARBUCKS",
    "AMAZON",
    "TESCO",
    "SAINSBURYS",
    "UBER",
    "SHELL",
    "BP",
    "QUICKBOOKS",
    "XERO",
    "SLACK",
    "MICROSOFT",
    "GOOGLE",
    "AWS",
    "DIGITALOCEAN",
    "STRIPE",
    "GOCARDLESS",
    "RENT",
    "UTILITIES",
    "INSURANCE",
    "ACCOUNTANT",
    "LEGAL",
    "COURIER",
    "CATERING",
    "STATIONERY",
    "TRAVEL",
]


def get_canonical_dictionary() -> List[str]:
    """Return the canonical merchant dictionary."""
    return CANONICAL_MERCHANTS.copy()
