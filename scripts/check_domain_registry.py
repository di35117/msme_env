"""
No-dependency validation for domain registry wiring.
Useful in environments where pytest is not installed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains import DOMAIN_REGISTRY, get_adapter


def main() -> None:
    if "msme_startup" not in DOMAIN_REGISTRY:
        raise SystemExit("FAIL: 'msme_startup' missing from DOMAIN_REGISTRY")
    adapter = get_adapter("msme_startup")
    if adapter.domain_id != "msme_startup":
        raise SystemExit("FAIL: adapter domain_id mismatch")
    if adapter.time_horizon != 36:
        raise SystemExit("FAIL: expected time_horizon=36")
    if adapter.total_entities != 30:
        raise SystemExit("FAIL: expected total_entities=30")
    if "wait_and_observe" not in adapter.valid_actions:
        raise SystemExit("FAIL: wait_and_observe not found in valid_actions")
    print("PASS: domain registry wiring is valid")


if __name__ == "__main__":
    main()

