# monosim/cli.py
from __future__ import annotations

import argparse
import csv
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .Simulator import (
    run_batch,
    KNOWN_STRATEGIES,
    generate_all_lineups,
    lineup_tag,
    Simulation,
    HumanController,
    StrategyController,
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

    # Canonical order so seating doesn't matter
    strategies: List[str] = []
    for strat in sorted(counts.keys()):
        strategies.extend([strat] * counts[strat])
    return strategies


def plot_winrates(win_rates: dict, outfile: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")  # avoid Tk/Tcl thread issues on Windows
    import matplotlib.pyplot as plt

    strategies = sorted(win_rates.keys())
    values = [win_rates[s] for s in strategies]

    wrapped = "\n".join(textwrap.wrap(title, width=40))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(strategies, values)
    plt.ylabel("Win rate (%)")
    plt.title(wrapped, pad=12)

    ymax = max(values) if values else 1
    plt.ylim(0, ymax * 1.25 if ymax > 0 else 1)

    # value labels
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.03,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_sweep_heatmap(rows: list[dict], outfile: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return

    strategies = sorted(KNOWN_STRATEGIES)

    def diversity_key(tag: str) -> tuple[int, str]:
        nonzero = tag.count("=")
        return (-nonzero, tag)

    lineup_tags = [r["lineup_tag"] for r in rows]
    lineup_tags = sorted(set(lineup_tags), key=diversity_key)

    tag_to_row = {r["lineup_tag"]: r for r in rows if r["lineup_tag"] in set(lineup_tags)}
    data = []
    for t in lineup_tags:
        r = tag_to_row[t]
        data.append([float(r[f"win_{s}"]) for s in strategies])

    wrapped = "\n".join(textwrap.wrap(title, width=50))

    plt.figure(figsize=(max(7, len(strategies) * 1.6),
                        max(6, len(lineup_tags) * 0.35)))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, label="Win rate (%)")
    plt.xticks(range(len(strategies)), strategies, rotation=30, ha="right")
    plt.yticks(range(len(lineup_tags)), lineup_tags)
    plt.title(wrapped, pad=14)

    plt.tight_layout(rect=(0, 0, 1, 0.90))
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

    # sanitize for Windows filenames (forbids <>:"/\|?* and control chars)
    safe_tag = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", tag).strip().strip("._") or "TAG"

    filename = f"{ts}__{safe_tag}__g{games}__t{max_turns}__cmin{cautious_min}__{seed_tag}.png"
    return str(Path(out_dir) / filename)


def _run_play_game(sim: Simulation, max_turns: int, step: bool) -> str:
    """
    Run an interactive match using the existing engine, but from the CLI so we can 'step'.
    Returns winner label.
    """
    for t in range(max_turns):
        sim.turn_count += 1
        print(f"\n=== Turn {sim.turn_count} ===")

        for i in range(4):
            p = sim.players[i]
            if p.balance <= 0:
                continue

            print(f"\n-- {p.name}'s turn (balance=${p.balance}) --")
            sim.run_turn(i)

            active = [pl for pl in sim.players if pl.balance > 0]
            if len(active) <= 1:
                winner = active[0] if active else max(sim.players, key=lambda x: x.balance)
                return sim.strategies[sim.players.index(winner)]

            if step:
                raw = input("\n[Enter]=continue | s=status | q=quit : ").strip().lower()
                if raw == "q":
                    return "QUIT"
                if raw == "s":
                    print("\nStatus:")
                    for pl in sim.players:
                        print(f"  {pl.name:22s} bal=${pl.balance:5d} pos={pl.position:2d} props={len(pl.properties)}")

    winner = max(sim.players, key=lambda p: p.balance)
    return sim.strategies[sim.players.index(winner)]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="monosim",
        description="Monopoly simulation + interactive play mode.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--lineup",
        nargs="+",
        help="(Batch) Strategy=Count entries that sum to 4. Example: --lineup Aggressive=2 Cautious=2",
    )
    mode.add_argument(
        "--sweep",
        action="store_true",
        help="(Batch) Run every not-yet-tested playstyle permutation (counts sum to 4; seating ignored).",
    )
    mode.add_argument(
        "--play",
        action="store_true",
        help="(Interactive) Play a match in the terminal against AI.",
    )

    parser.add_argument("--games", type=int, default=1000, help="(Batch) Number of games to simulate PER lineup.")
    parser.add_argument("--max-turns", type=int, default=1000, help="Max turns per game.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible runs.")
    parser.add_argument(
        "--cautious-min",
        type=int,
        default=500,
        help="Cautious players only buy if (balance - price) >= this value.",
    )
    parser.add_argument("--no-rich", action="store_true", help="Disable Rich progress bar (batch only).")

    # plotting
    parser.add_argument("--plot", action="store_true", help="Save a bar chart of win rates.")
    parser.add_argument("--out-dir", default="plots", help="Folder to save plot PNGs.")
    parser.add_argument("--auto-name", action="store_true", help="Auto-generate a unique plot filename.")
    parser.add_argument("--summary-plot", action="store_true", help="(Sweep only) Also generate one heatmap PNG.")

    # sweep persistence
    parser.add_argument(
        "--results-file",
        default="results/sweep_results.csv",
        help="(Sweep only) CSV to append results to; used to skip already-tested lineups.",
    )
    parser.add_argument("--force", action="store_true", help="(Sweep only) Re-run lineups even if already in CSV.")

    # interactive play options
    parser.add_argument("--human-index", type=int, default=1, help="(Play) Which seat is human: 1-4.")
    parser.add_argument("--human-name", type=str, default="Human", help="(Play) Display name for the human player.")
    parser.add_argument(
        "--ai",
        nargs="+",
        default=None,
        help="(Play) Three AI strategies for the non-human players (e.g. --ai Aggressive Cautious ColorCollector).",
    )
    parser.add_argument("--step", action="store_true", help="(Play) Pause after each player's turn.")

    args = parser.parse_args(argv)
    params = {"cautious_min": args.cautious_min}

    # -------------------- PLAY MODE --------------------
    if args.play:
        hi = args.human_index
        if hi < 1 or hi > 4:
            raise SystemExit("--human-index must be 1..4")

        if args.ai is None:
            ai_list = ["Aggressive", "Cautious", "ColorCollector"]  # default 3 AIs
        else:
            ai_list = args.ai

        if len(ai_list) != 3:
            raise SystemExit("(Play) --ai must specify exactly 3 strategies.")

        for s in ai_list:
            if s not in KNOWN_STRATEGIES:
                raise SystemExit(f"(Play) Unknown AI strategy '{s}'. Known: {sorted(KNOWN_STRATEGIES)}")

        human_zero = hi - 1

        controllers = []
        ai_i = 0
        for seat in range(4):
            if seat == human_zero:
                controllers.append(HumanController(name=args.human_name))
            else:
                controllers.append(StrategyController(ai_list[ai_i]))
                ai_i += 1

        sim = Simulation(
            controllers=controllers,
            seed=args.seed,
            params=params,
            verbose=False,  # HumanController prints prompts; keep engine spam down
        )

        print("\nStarting interactive match!")
        print("Seats:")
        for i, c in enumerate(controllers, start=1):
            print(f"  Seat {i}: {c.label()}")
        print(f"\nMax turns: {args.max_turns} | Seed: {args.seed} | Cautious-min: {args.cautious_min}")

        winner = _run_play_game(sim, max_turns=args.max_turns, step=args.step)
        print(f"\nWinner: {winner}")
        return

    # -------------------- LINEUP (BATCH) MODE --------------------
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
            plot_path = (
                make_plot_path(args.out_dir, tag, args.games, args.max_turns, args.cautious_min, args.seed)
                if args.auto_name
                else str(Path(args.out_dir) / "winrates.png")
            )
            plot_winrates(win_rates, plot_path, title=f"Win Rates ({tag})")
            print(f"Saved plot to: {plot_path}")

        return

    # -------------------- SWEEP (BATCH) MODE --------------------
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


if __name__ == "__main__":
    main()
