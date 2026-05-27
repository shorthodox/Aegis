"""
generate_dev_code.py - Developer key generator for Aegis.

Generates a cryptographically secure, random 32-character alphanumeric
developer key. Import and call `generate_dev_key()` from main.py or any
other module that needs on-demand key generation.
"""

import secrets
import string

_ALPHABET = string.ascii_letters + string.digits  # A-Z, a-z, 0-9


def generate_dev_key(length: int = 32) -> str:
    """Return a cryptographically secure random alphanumeric developer key.

    Args:
        length: Number of characters in the generated key (default: 32).

    Returns:
        A random alphanumeric string of the requested length, e.g.
        ``"aB3xQ7mNpL2wRtYv8KdZeUcFjHsOiG1n"``.
    """
    return "".join(secrets.choice(_ALPHABET) for _ in range(length))
