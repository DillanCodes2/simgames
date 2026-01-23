# monosim/Simulator.py
from __future__ import annotations

import random
from collections import Counter
from typing import List, Dict, Tuple, Optional

from .Board import Board, Space
from .Player import Player

KNOWN_STRATEGIES = {"Aggressive", "Cautious", "RailRoadTycoon", "ColorCollector"}


class Simulation:
    def __init__(self, strategies: List[str], seed: Optional[int] = None, params: Optional[dict] = None):
        if len(strategies) != 4:
            raise ValueError("Simulation requires exactly 4 players/strategies.")
        unknown = [s for s in strategies if s not in KNOWN_STRATEGIES]
        if unknown:
            raise ValueError(f"Unknown strategy(ies): {unknown}. Known: {sorted(KNOWN_STRATEGIES)}")

        self.params = params or {}
        self.rng = random.Random(seed)
        self.board = Board()
        self.players = [Player(f"P{i+1} ({strat})") for i, strat in enumerate(strategies)]
        self.strategies = strategies
        self.turn_count = 0
        self.last_roll = 0

    def roll_dice(self) -> int:
        return self.rng.randint(1, 6) + self.rng.randint(1, 6)

    def calculate_rent(self, space: Space) -> int:
        if space.type == "property":
            if space.owner and space.owner.owns_monopoly(space.color, self.board) and space.houses == 0:
                return space.base_rent * 2
            if space.houses > 0:
                return space.base_rent * (5 ** space.houses)
            return space.base_rent

        if space.type == "railroad":
            owner = space.owner
            if not owner:
                return space.base_rent
            rr_owned = sum(1 for p in owner.properties if p.type == "railroad")
            return space.base_rent * rr_owned

        if space.type == "utility":
            owner = space.owner
            if not owner:
                return 0
            util_owned = sum(1 for p in owner.properties if p.type == "utility")
            multiplier = 4 if util_owned == 1 else 10
            return multiplier * self.last_roll

        return 0

    def handle_property_logic(self, player: Player, space: Space, strategy: str) -> str:
        if space.type in {"property", "railroad", "utility"} and space.owner is None:
            if strategy == "Aggressive":
                player.buy_property(space)

            elif strategy == "Cautious":
                cautious_min = int(self.params.get("cautious_min", 500))
                if player.balance - space.price >= cautious_min:
                    player.buy_property(space)

            elif strategy == "RailRoadTycoon":
                if space.type in {"railroad", "utility"}:
                    player.buy_property(space)
                elif player.balance - space.price >= 700:
                    player.buy_property(space)

            elif strategy == "ColorCollector":
                already_owns_color = space.color is not None and any(p.color == space.color for p in player.properties)
                if already_owns_color or (player.balance - space.price >= 400):
                    player.buy_property(space)

            return "OK"

        if space.owner and space.owner is not player:
            rent = self.calculate_rent(space)
            success = player.remove_funds(rent)
            space.owner.add_funds(rent)
            return "OK" if success else "BANKRUPT"

        return "OK"

    def handle_housing_logic(self, player: Player, strategy: str) -> None:
        for prop in player.properties:
            if prop.type != "property" or prop.color is None:
                continue
            if not player.owns_monopoly(prop.color, self.board):
                continue
            if prop.houses >= 5:
                continue

            if strategy == "Aggressive" and player.balance > prop.house_price:
                player.remove_funds(prop.house_price)
                prop.houses += 1
            elif strategy == "Cautious" and (player.balance - prop.house_price) > 1000:
                player.remove_funds(prop.house_price)
                prop.houses += 1

    def run_turn(self, player_index: int) -> bool:
        player = self.players[player_index]
        strategy = self.strategies[player_index]

        if player.balance <= 0:
            return False

        self.last_roll = self.roll_dice()
        player.move(self.last_roll)
        current_space = self.board.get_space(player.position)
        status = self.handle_property_logic(player, current_space, strategy)

        if status != "BANKRUPT":
            self.handle_housing_logic(player, strategy)

        return status != "BANKRUPT"

    def run_full_game(self, max_turns: int = 1000) -> Tuple[str, List[Tuple[str, int, int]]]:
        for _ in range(max_turns):
            self.turn_count += 1
            for i in range(4):
                self.run_turn(i)

            active_players = [p for p in self.players if p.balance > 0]
            if len(active_players) <= 1:
                break

        winner = max(self.players, key=lambda p: p.balance)
        winner_strategy = self.strategies[self.players.index(winner)]
        results = [(p.name, p.balance, len(p.properties)) for p in self.players]
        return winner_strategy, results


def run_batch(
    iterations: int,
    strategies: List[str],
    max_turns: int = 1000,
    seed: Optional[int] = None,
    params: Optional[dict] = None,
    use_rich: bool = True,
) -> Dict[str, float]:
    if len(strategies) != 4:
        raise ValueError("run_batch requires exactly 4 strategies.")

    rng = random.Random(seed)
    wins = Counter()

    # Try Rich progress if requested; otherwise silently fall back.
    progress = None
    task_id = None
    if use_rich:
        try:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                BarColumn,
                TextColumn,
                MofNCompleteColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,  # cleans up the bar when done
            )
            progress.start()
            task_id = progress.add_task("Simulating games", total=iterations)
        except Exception:
            progress = None
            task_id = None

    try:
        for _ in range(iterations):
            game_seed = rng.randint(0, 2**31 - 1)
            sim = Simulation(strategies, seed=game_seed, params=params)
            winner_strategy, _ = sim.run_full_game(max_turns=max_turns)
            wins[winner_strategy] += 1

            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)
    finally:
        if progress is not None:
            progress.stop()

    return {s: (wins[s] / iterations) * 100.0 for s in KNOWN_STRATEGIES}
