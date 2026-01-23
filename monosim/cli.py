# monosim/cli.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Iterable

from .Simulator import (
    run_batch,
    KNOWN_STRATEGIES,
    generate_all_lineups,
    lineup_tag,
)


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

    # Canonical order to avoid treating seating as different.
    strategies: List[str] = []
    for strat in sorted(counts.keys()):
        strategies.extend([strat] * counts[strat])
    return strategies


def plot_winrates(win_rates: dict, outfile: str, title: str) -> None:
    import matplotlib.pyplot as plt

    strategies = sorted(win_rates.keys())
    values = [win_rates[s] for s in strategies]

    plt.figure()
    plt.bar(strategies, values)
    plt.ylabel("Win rate (%)")
    plt.title(title)
    plt.ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_sweep_heatmap(rows: list[dict], outfile: str, title: str) -> None:
    """
    Heatmap: lineups (rows) x strategies (cols), colored by win%.
    rows is a list of dicts produced by sweep CSV writing.
    """
    import matplotlib.pyplot as plt

    if not rows:
        return

    strategies = sorted(KNOWN_STRATEGIES)

    # Keep lineups in deterministic order for readability:
    # first by "more diverse" (more nonzero counts), then lexicographically.
    def diversity_key(tag: str) -> tuple[int, str]:
        nonzero = tag.count("=")  # rough; tag already excludes zeros
        return (-nonzero, tag)

    lineup_tags = [r["lineup_tag"] for r in rows]
    lineup_tags = sorted(set(lineup_tags), key=diversity_key)

    # Build matrix in same order.
    tag_to_row = {r["lineup_tag"]: r for r in rows if r["lineup_tag"] in set(lineup_tags)}
    data = []
    for t in lineup_tags:
        r = tag_to_row[t]
        data.append([float(r[f"win_{s}"]) for s in strategies])

    plt.figure(figsize=(max(7, len(strategies) * 1.6), max(6, len(lineup_tags) * 0.35)))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, label="Win rate (%)")
    plt.xticks(range(len(strategies)), strategies, rotation=30, ha="right")
    plt.yticks(range(len(lineup_tags)), lineup_tags)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def load_existing_tags(results_file: str) -> set[str]:
    path = Path(results_file)
    if not path.exists():
        return set()
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {row.get("lineup_tag", "") for row in reader if row.get("lineup_tag")}
    except Exception:
        return set()


def append_result_row(results_file: str, row: dict) -> None:
    path = Path(results_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(row.keys())
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_plot_path(
    out_dir: str,
    tag: str,
    games: int,
    max_turns: int,
    cautious_min: int,
    seed: int | None,
) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    seed_tag = f"seed{seed}" if seed is not None else "seedNone"
    filename = f"{ts}__{tag}__g{games}__t{max_turns}__cmin{cautious_min}__{seed_tag}.png"
    return str(Path(out_dir) / filename)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="monosim",
        description="Run autonomous Monopoly simulations to estimate win rates by playstyle.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--lineup",
        nargs="+",
        help="Space-separated Strategy=Count entries that sum to 4. Example: --lineup Aggressive=2 Cautious=2",
    )
    mode.add_argument(
        "--sweep",
        action="store_true",
        help="Run every (not-yet-tested) playstyle permutation (counts sum to 4; seating ignored).",
    )

    parser.add_argument("--games", type=int, default=1000, help="Number of games to simulate PER lineup.")
    parser.add_argument("--max-turns", type=int, default=1000, help="Max turns per game.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible runs.")

    parser.add_argument(
        "--cautious-min",
        type=int,
        default=500,
        help="Cautious players only buy if (balance - price) >= this value. Ignored if no Cautious players.",
    )

    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable the Rich progress bar.",
    )

    # plotting
    parser.add_argument("--plot", action="store_true", help="Save a bar chart of win rates.")
    parser.add_argument("--out-dir", default="plots", help="Folder to save plot PNGs.")
    parser.add_argument(
        "--auto-name",
        action="store_true",
        help="Auto-generate a unique plot filename.",
    )
    parser.add_argument(
        "--summary-plot",
        action="store_true",
        help="(Sweep only) Also generate one heatmap PNG summarizing all completed lineups in this sweep run.",
    )

    # sweep persistence / skipping
    parser.add_argument(
        "--results-file",
        default="results/sweep_results.csv",
        help="(Sweep only) CSV file to append results to; used to skip already-tested lineups.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="(Sweep only) Re-run lineups even if they already appear in --results-file.",
    )

    args = parser.parse_args(argv)
    params = {"cautious_min": args.cautious_min}

    if args.lineup:
        strategies = parse_lineup(args.lineup)
        tag = lineup_tag(strategies)

        win_rates = run_batch(
            args.games,
            strategies,
            max_turns=args.max_turns,
            seed=args.seed,
            params=params,
            use_rich=(not args.no_rich),
        )

        print("Lineup:", strategies)
        print(f"Tag: {tag}")
        print(
            f"Games: {args.games} | Max turns: {args.max_turns} | Seed: {args.seed} | "
            f"Cautious-min: {args.cautious_min}"
        )
        print("Win rates (%):")
        for strat in sorted(KNOWN_STRATEGIES):
            print(f"  {strat:14s} {win_rates[strat]:6.2f}")

        if args.plot:
            plot_path = make_plot_path(
                args.out_dir,
                tag,
                args.games,
                args.max_turns,
                args.cautious_min,
                args.seed,
            ) if args.auto_name else str(Path(args.out_dir) / "winrates.png")

            plot_winrates(win_rates, plot_path, title=f"Win Rates ({tag})")
            print(f"Saved plot to: {plot_path}")

        return

    # --- SWEEP MODE ---
    existing = load_existing_tags(args.results_file) if (not args.force) else set()
    all_lineups = list(generate_all_lineups(total_players=4))
    remaining = [ls for ls in all_lineups if (lineup_tag(ls) not in existing)]

    print(f"Known strategies: {sorted(KNOWN_STRATEGIES)}")
    print(f"Total unique playstyle permutations (counts sum to 4; seating ignored): {len(all_lineups)}")
    print(f"Already in results file: {len(existing)}")
    print(f"To run now: {len(remaining)}")
    print(f"Results CSV: {args.results_file}")

    completed_rows: list[dict] = []

    for idx, strategies in enumerate(remaining, start=1):
        tag = lineup_tag(strategies)

        # If user supplies --seed, vary it per lineup deterministically for reproducibility.
        run_seed = (args.seed + (idx - 1)) if args.seed is not None else None

        print(f"\n[{idx}/{len(remaining)}] Running lineup {tag}: {strategies}")

        win_rates = run_batch(
            args.games,
            strategies,
            max_turns=args.max_turns,
            seed=run_seed,
            params=params,
            use_rich=(not args.no_rich),
        )

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "lineup_tag": tag,
            "games": args.games,
            "max_turns": args.max_turns,
            "cautious_min": args.cautious_min,
            "seed": run_seed if run_seed is not None else "",
        }
        for s in sorted(KNOWN_STRATEGIES):
            row[f"win_{s}"] = f"{win_rates[s]:.4f}"

        append_result_row(args.results_file, row)
        completed_rows.append(row)

        if args.plot:
            plot_path = make_plot_path(
                args.out_dir,
                tag,
                args.games,
                args.max_turns,
                args.cautious_min,
                run_seed,
            )
            plot_winrates(win_rates, plot_path, title=f"Win Rates ({tag})")
            print(f"Saved plot to: {plot_path}")

    if args.summary_plot and completed_rows:
        heatmap_path = make_plot_path(
            args.out_dir,
            tag="SWEEP_HEATMAP",
            games=args.games,
            max_turns=args.max_turns,
            cautious_min=args.cautious_min,
            seed=args.seed,
        )
        plot_sweep_heatmap(
            completed_rows,
            heatmap_path,
            title=f"Sweep Heatmap (g={args.games}, t={args.max_turns}, cmin={args.cautious_min})",
        )
        print(f"\nSaved sweep heatmap to: {heatmap_path}")

    print("\nDone.")