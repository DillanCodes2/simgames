# monosim/cli.py
from __future__ import annotations

import argparse
from typing import Dict, List, Optional

from .Simulator import run_batch, KNOWN_STRATEGIES


def parse_lineup(items: List[str]) -> List[str]:
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


def plot_winrates(win_rates: dict, outfile: str) -> None:
    import matplotlib.pyplot as plt

    strategies = sorted(win_rates.keys())
    values = [win_rates[s] for s in strategies]

    plt.figure()
    plt.bar(strategies, values)
    plt.ylabel("Win rate (%)")
    plt.title("Win Rates by Strategy (this lineup)")
    plt.ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

from pathlib import Path
from datetime import datetime

def make_plot_path(out_dir: str, strategies: list[str], games: int, max_turns: int, cautious_min: int, seed: int | None) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # lineup tag like A2_C2 or A1_C1_R1_CC1 (short but readable)
    counts = {}
    for s in strategies:
        counts[s] = counts.get(s, 0) + 1
    tag = "_".join(f"{k[:2]}{v}" for k, v in sorted(counts.items()))  # e.g. Ag2_Ca2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    seed_tag = f"seed{seed}" if seed is not None else "seedNone"

    filename = f"{ts}__{tag}__g{games}__t{max_turns}__cmin{cautious_min}__{seed_tag}.png"
    return str(Path(out_dir) / filename)


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
    parser.add_argument("--max-turns", type=int, default=1000, help="Max turns per game.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible batch runs.")

    # ✅ only defined ONCE
    parser.add_argument(
        "--cautious-min",
        type=int,
        default=500,
        help="Cautious players only buy if (balance - price) >= this value. Ignored if no Cautious players.",
    )

    # optional: show progress so big runs don't look frozen
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N games (0 disables).",
    )

    parser.add_argument(
    "--no-rich",
    action="store_true",
    help="Disable the Rich progress bar.",
)


    parser.add_argument("--plot", action="store_true", help="Save a bar chart of win rates.")
    parser.add_argument("--plot-file", default="winrates.png", help="Output PNG filename for the chart.")
    parser.add_argument("--out-dir", default="plots", help="Folder to save plot PNGs.")
    parser.add_argument("--auto-name", action="store_true",
                        help="Auto-generate a unique plot filename (ignores --plot-file).")
    parser.add_argument("--runs", type=int, default=1,
                        help="How many independent batch runs to execute.")

    args = parser.parse_args(argv)
    strategies = parse_lineup(args.lineup)

    params = {"cautious_min": args.cautious_min}

    for run_i in range(1, args.runs + 1):
        # Use different seed per run unless user pins one
        run_seed = args.seed if args.seed is not None else None

        win_rates = run_batch(
            args.games,
            strategies,
            max_turns=args.max_turns,
            seed=args.seed,
            params=params,
            use_rich=(not args.no_rich),
)


    print(f"\n=== Run {run_i}/{args.runs} ===")
    print("Lineup:", strategies)
    print(
        f"Games: {args.games} | Max turns: {args.max_turns} | Seed: {run_seed} | "
        f"Cautious-min: {args.cautious_min}"
    )
    print("Win rates (%):")
    for strat in sorted(KNOWN_STRATEGIES):
        print(f"  {strat:14s} {win_rates[strat]:6.2f}")

    if args.plot:
        if args.auto_name:
            plot_path = make_plot_path(args.out_dir, strategies, args.games, args.max_turns, args.cautious_min, run_seed)
        else:
            # still allow manual naming if you want
            from pathlib import Path
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            plot_path = str(Path(args.out_dir) / args.plot_file)

        plot_winrates(win_rates, plot_path)
        print(f"Saved plot to: {plot_path}")

    print("\nLineup:", strategies)
    print(
        f"Games: {args.games} | Max turns: {args.max_turns} | Seed: {args.seed} | "
        f"Cautious-min: {args.cautious_min}"
    )
    print("\nWin rates (%):")
    for strat in sorted(KNOWN_STRATEGIES):
        print(f"  {strat:14s} {win_rates[strat]:6.2f}")

    if args.plot:
        plot_winrates(win_rates, args.plot_file)
        print(f"\nSaved plot to: {args.plot_file}")


if __name__ == "__main__":
    main()
