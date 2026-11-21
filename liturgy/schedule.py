"""Pure host-side digital skip scheduling; NOT quantum many-body scarring."""

from math import isqrt


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % divisor != 0 for divisor in range(2, isqrt(n) + 1))


def skip_indices(depth: int, prime: int, event_count: int) -> tuple[int, ...]:
    """Return exactly event_count positive prime-spaced indices < depth.

    No hardware operations occur. The returned indices are suitable only as
    an input to a separately validated and bounded hardware scheduler.
    """
    for name, value in (("depth", depth), ("prime", prime), ("event_count", event_count)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if depth < 1 or not _is_prime(prime) or event_count < 0:
        raise ValueError("Require depth >= 1, a prime spacing >= 2, and event_count >= 0")
    if event_count > (depth - 1) // prime:
        raise ValueError("Requested skip events do not fit within depth")
    return tuple(prime * i for i in range(1, event_count + 1))
