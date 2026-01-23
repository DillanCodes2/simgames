# monosim/cli.py
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple

# Simulator + strategies
from .Simulator import (
    KNOWN_STRATEGIES,
    STRATEGY_ORDER,
    Simulation,
    StrategyController,
    HumanController,
)

# Optional GUI imports (only needed for --play-gui)
try:
    from .ui_pygame import PygameUI
    from .gui_controller import PygameHumanController
except Exception:
    PygameUI = None
    PygameHumanController = None


# ----------------------------
# Helpers: filenames / tags
# ----------------------------

_WINDOWS_FORBIDDEN = r'<>:"/\\|?*'
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_filename_component(s: str, replacement: str = "_") -> str:
    """Make a string safe as a filename component (Windows-safe)."""
    # Replace forbidden chars
    s2 = "".join((ch if ch not in _WINDOWS_FORBIDDEN else replacement) for ch in s)

    # Strip control chars
    s2 = "".join((ch if (ord(ch) >= 32) else replacement) for ch in s2)

    # Collapse whitespace
    s2 = re.sub(r"\s+", " ", s2).strip()

    # No trailing dots/spaces on Windows
    s2 = s2.rstrip(" .")

    if not s2:
        s2 = "plot"

    # Avoid reserved names
    root = s2.split(".")[0].upper()
    if root in _WINDOWS_RESERVED:
        s2 = f"_{s2}"

    # Reasonable length cap
    if len(s2) > 140:
        s2 = s2[:140].rstrip(" .")

    return s2


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def lineup_tag(strategies: List[str]) -> str:
    """
    Canonical tag for a lineup, ignoring seating:
      ['Aggressive','Cautious','Cautious','ColorCollector']
        -> 'Aggressive=1|Cautious=2|ColorCollector=1'
    """
    counts = Counter(strategies)
    parts = []
    for s in STRATEGY_ORDER:
        if counts.get(s, 0) > 0:
            parts.append(f"{s}={counts[s]}")
    return "|".join(parts) if parts else "EMPTY"


def parse_lineup(args_lineup: List[str]) -> List[str]:
    # Validate
    for s in args_lineup:
        if s not in KNOWN_STRATEGIES:
            raise SystemExit(f"Unknown strategy '{s}'. Known: {sorted(KNOWN_STRATEGIES)}")
    return list(args_lineup)


def generate_all_lineups(total_players: int = 4) -> Iterable[List[str]]:
    """
    Generate all unique playstyle permutations (counts sum to total_players; seating ignored).
    With 4 strategies and 4 players => C(4+4-1,4)=35 unique count-combos.
    """
    strategies = list(STRATEGY_ORDER)

    def rec(i: int, remaining: int, current: List[str]):
        if i == len(strategies) - 1:
            # last strategy gets the rest
            out = current + [strategies[i]] * remaining
            yield out
            return
        for k in range(remaining + 1):
            yield from rec(i + 1, remaining - k, current + [strategies[i]] * k)

    yield from rec(0, total_players, [])


# ----------------------------
# Plotting helpers
# ----------------------------

def plot_winrates(win_rates: Dict[str, float], outfile: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap

    strategies = sorted(win_rates.keys())
    values = [win_rates[s] for s in strategies]

    wrapped = "\n".join(textwrap.wrap(title, width=40))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(strategies, values)

    plt.ylabel("Win rate (%)")
    plt.title(wrapped, pad=12)

    ymax = max(values) if values else 1
    plt.ylim(0, ymax * 1.25 if ymax > 0 else 1)

    # Value labels on bars
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.03,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Reserve space for title
    plt.tight_layout(rect=(0, 0, 1, 0.90))

    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_sweep_heatmap(rows: list[dict], outfile: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap

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

    plt.figure(figsize=(max(7, len(strategies) * 1.6), max(6, len(lineup_tags) * 0.35)))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, label="Win rate (%)")
    plt.xticks(range(len(strategies)), strategies, rotation=30, ha="right")
    plt.yticks(range(len(lineup_tags)), lineup_tags)
    plt.title(wrapped, pad=14)

    plt.tight_layout(rect=(0, 0, 1, 0.90))

    plt.savefig(outfile, dpi=200)
    plt.close()


# ----------------------------
# Results CSV helpers
# ----------------------------

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


# ----------------------------
# CLI
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="monosim")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lineup", nargs="+", help="Run exactly one lineup (4 strategies).")
    mode.add_argument("--sweep", action="store_true", help="Run all untested playstyle permutations.")
    mode.add_argument("--play", action="store_true", help="Interactive terminal play (human + AIs).")
    mode.add_argument("--play-gui", action="store_true", help="Interactive GUI play using pygame.")

    p.add_argument("--games", type=int, default=100, help="Number of games per run / lineup.")
    p.add_argument("--max-turns", type=int, default=1000, help="Max turns per game.")
    p.add_argument("--seed", type=int, default=None, help="Base RNG seed.")
    p.add_argument("--cautious-min", type=int, default=500, help="Cautious starts building after this cash.")
    p.add_argument("--no-rich", action="store_true", help="Disable the 'rich' reward in simulation mode (if present).")

    # Plotting + output
    p.add_argument("--plot", action="store_true", help="Save plot(s).")
    p.add_argument("--out-dir", default="plots", help="Directory to write plots into.")
    p.add_argument("--plot-file", default=None, help="Override output plot filename.")
    p.add_argument("--auto-name", action="store_true", help="Auto-name plot files.")
    p.add_argument("--summary-plot", action="store_true", help="After a sweep, also write a heatmap summary plot.")
    p.add_argument("--results-file", default="results/sweep_results.csv", help="CSV for sweep results.")
    p.add_argument("--force", action="store_true", help="Re-run even if lineup already exists in results file.")

    # Play mode options
    p.add_argument("--human-index", type=int, default=0, help="Which seat is the human (0-based).")
    p.add_argument("--human-name", default="Human", help="Human display name.")
    p.add_argument("--ai", nargs="*", default=[], help="AI strategies for the non-human players.")
    p.add_argument("--step", action="store_true", help="(Play/GUI) click/enter to advance turns (UI-dependent).")

    return p


def _make_plot_path(out_dir: str, tag: str, games: int, max_turns: int, seed: Optional[int]) -> Path:
    out = ensure_dir(out_dir)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = sanitize_filename_component(tag)
    safe_seed = "None" if seed is None else str(seed)
    fname = f"{stamp}__{safe_tag}__g{games}__t{max_turns}__seed{safe_seed}.png"
    return out / fname


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    # Normalize out dir early so everything goes into /plots (or whatever you pass)
    ensure_dir(args.out_dir)

    # --- SINGLE LINEUP MODE ---
    if args.lineup is not None:
        strategies = parse_lineup(args.lineup)
        if len(strategies) != 4:
            raise SystemExit(f"--lineup must contain exactly 4 strategies, got {len(strategies)}")

        tag = lineup_tag(strategies)

        sim = Simulation(
            strategies=strategies,
            seed=args.seed,
            max_turns=args.max_turns,
            cautious_min=args.cautious_min,
            rich=(not args.no_rich),
        )
        win_rates = sim.run_games(num_games=args.games)

        if args.plot:
            outfile = args.plot_file or str(_make_plot_path(args.out_dir, tag, args.games, args.max_turns, args.seed))
            plot_winrates(win_rates, outfile, title=f"Win Rates ({tag})")
            print(f"Saved plot to: {outfile}")

        print(win_rates)
        return

    # --- SWEEP MODE ---
    if args.sweep:
        existing = load_existing_tags(args.results_file) if (not args.force) else set()
        all_lineups = list(generate_all_lineups(total_players=4))
        remaining = [ls for ls in all_lineups if (lineup_tag(ls) not in existing)]

        print(f"Known strategies: {sorted(KNOWN_STRATEGIES)}")
        print(f"Total unique playstyle permutations (counts sum to 4; seating ignored): {len(all_lineups)}")
        print(f"Already in results file: {len(existing)}")
        print(f"To run now: {len(remaining)}")
        print(f"Results CSV: {args.results_file}")
        print()

        completed_rows: list[dict] = []

        for idx, strategies in enumerate(remaining, start=1):
            tag = lineup_tag(strategies)

            # If user supplies --seed, vary it per lineup to avoid identical game streams
            seed = None if args.seed is None else (args.seed + idx * 9973)

            print(f"[{idx}/{len(remaining)}] Running lineup {tag}: {strategies}")

            sim = Simulation(
                strategies=strategies,
                seed=seed,
                max_turns=args.max_turns,
                cautious_min=args.cautious_min,
                rich=(not args.no_rich),
            )
            win_rates = sim.run_games(num_games=args.games)

            row = {
                "lineup_tag": tag,
                "games": args.games,
                "max_turns": args.max_turns,
                "seed": seed,
                **{f"win_{s}": win_rates.get(s, 0.0) for s in sorted(KNOWN_STRATEGIES)},
            }
            append_result_row(args.results_file, row)
            completed_rows.append(row)

            if args.plot:
                outfile = args.plot_file or str(_make_plot_path(args.out_dir, tag, args.games, args.max_turns, seed))
                plot_winrates(win_rates, outfile, title=f"Win Rates ({tag})")
                print(f"Saved plot to: {outfile}")

            print()

        # Summary heatmap
        if args.summary_plot and completed_rows:
            out = ensure_dir(args.out_dir)
            outfile = out / f"{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}__sweep_heatmap.png"
            plot_sweep_heatmap(completed_rows, str(outfile), title="Sweep Heatmap (win rates by lineup)")
            print(f"Saved sweep heatmap to: {outfile}")

        return

    # --- PLAY (terminal) MODE ---
    if args.play:
        # Build controllers list (4 players total)
        if len(args.ai) != 3:
            raise SystemExit("--play requires exactly 3 AI strategies via --ai (human + 3 AIs = 4 players).")

        for s in args.ai:
            if s not in KNOWN_STRATEGIES:
                raise SystemExit(f"Unknown AI strategy '{s}'. Known: {sorted(KNOWN_STRATEGIES)}")

        human_index = int(args.human_index)
        if not (0 <= human_index <= 3):
            raise SystemExit("--human-index must be in [0,3].")

        controllers = []
        ai_iter = iter(args.ai)
        for i in range(4):
            if i == human_index:
                controllers.append(HumanController(name=args.human_name))
            else:
                controllers.append(StrategyController(next(ai_iter)))

        sim = Simulation(
            controllers=controllers,
            seed=args.seed,
            max_turns=args.max_turns,
            cautious_min=args.cautious_min,
            rich=(not args.no_rich),
        )
        # Stash for any UIs that want it
        sim.human_index = human_index  # type: ignore[attr-defined]

        winner = sim.run_interactive(step=args.step)
        print(f"Winner: {winner}")
        return

    # --- PLAY (GUI) MODE ---
    if args.play_gui:
        if PygameUI is None or PygameHumanController is None:
            raise SystemExit(
                "GUI mode requires pygame. Install it and re-run.\n"
                "Recommended: pip install pygame-ce"
            )

        if len(args.ai) != 3:
            raise SystemExit("--play-gui requires exactly 3 AI strategies via --ai (human + 3 AIs = 4 players).")

        for s in args.ai:
            if s not in KNOWN_STRATEGIES:
                raise SystemExit(f"Unknown AI strategy '{s}'. Known: {sorted(KNOWN_STRATEGIES)}")

        human_index = int(args.human_index)
        if not (0 <= human_index <= 3):
            raise SystemExit("--human-index must be in [0,3].")

        # Create sim with placeholder controllers first (we need sim instance for UI)
        controllers = [StrategyController(args.ai[0]), StrategyController(args.ai[1]),
                       StrategyController(args.ai[2]), StrategyController(args.ai[2])]
        # We'll overwrite correctly below; this just ensures type.
        controllers = []

        sim = Simulation(
            controllers=[],  # set after UI is created
            seed=args.seed,
            max_turns=args.max_turns,
            cautious_min=args.cautious_min,
            rich=(not args.no_rich),
        )
        sim.human_index = human_index  # type: ignore[attr-defined]

        ui = PygameUI(sim)
        human = PygameHumanController(ui, name=args.human_name)

        ai_iter = iter(args.ai)
        controllers = []
        for i in range(4):
            if i == human_index:
                controllers.append(human)
            else:
                controllers.append(StrategyController(next(ai_iter)))

        sim.controllers = controllers  # type: ignore[attr-defined]

        # Let UI drive the loop
        winner = sim.run_interactive(step=args.step, ui=ui)
        ui.push_log(f"Winner: {winner}")
        ui.run_until_quit()
        return


if __name__ == "__main__":
    main()
