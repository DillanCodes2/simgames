# monosim/cli.py
from __future__ import annotations

import argparse
from typing import Dict, List, Optional

from .Simulator import run_batch, KNOWN_STRATEGIES


def parse_lineup(items: List[str]) -> List[str]:
    """Parses ['Aggressive=2','Cautious=2'] -> 4-length strategy list."""
    counts: Dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Bad format '{item}'. Use Strategy=Count, e.g. Aggressive=2"
            )
        strat, raw = item.split("=", 1)
        strat = strat.strip()
        if strat not in KNOWN_STRATEGIES:
            raise argparse.ArgumentTypeError(
                f"Unknown strategy '{strat}'. Known: {sorted(KNOWN_STRATEGIES)}"
            )
        try:
            n = int(raw.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(f"Count must be an int: '{item}'")
        if n < 0:
            raise argparse.ArgumentTypeError(f"Count must be >= 0: '{item}'")
        counts[strat] = counts.get(strat, 0) + n

    total = sum(counts.values())
    if total != 4:
        raise argparse.ArgumentTypeError(f"Counts must sum to 4 (got {total}).")

    strategies: List[str] = []
    for strat in sorted(counts.keys()):
        strategies.extend([strat] * counts[strat])
    return strategies


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="monosim",
        description="Run autonomous Monopoly simulations to estimate win rates by playstyle."
    )
    parser.add_argument(
        "--lineup",
        nargs="+",
        required=True,
        help="Space-separated Strategy=Count entries that sum to 4. Example: --lineup Aggressive=2 Cautious=2",
    )
    parser.add_argument("--games", type=int, default=1000, help="Number of games to simulate.")
    parser.add_argument("--max-turns", type=int, default=1000, help="Max turns per game before declaring highest balance winner.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible batch runs.")

    args = parser.parse_args(argv)
    strategies = parse_lineup(args.lineup)

    win_rates = run_batch(args.games, strategies, max_turns=args.max_turns, seed=args.seed)

    print("\nLineup:", strategies)
    print(f"Games: {args.games} | Max turns: {args.max_turns} | Seed: {args.seed}")
    print("\nWin rates (%):")
    for strat in sorted(KNOWN_STRATEGIES):
        print(f"  {strat:14s} {win_rates[strat]:6.2f}")


if __name__ == "__main__":
    main()
