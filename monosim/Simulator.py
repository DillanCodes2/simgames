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
        self.rng = random.Random()
        self.last_roll = 0  # keep for utilities later


    def roll_dice(self):
        d1 = self.rng.randint(1, 6)
        d2 = self.rng.randint(1, 6)
        return d1, d2, d1 + d2, (d1 == d2)

    JAIL_INDEX = 10        # "Just Visiting / In Jail" is index 10 on standard board
    GO_TO_JAIL_INDEX = 30  # "Go To Jail" is index 30

    def send_to_jail(self, player: Player):
        player.position = self.JAIL_INDEX
        player.is_in_jail = True
        player.jail_turns = 0

    def take_jail_turn(self, player: Player) -> tuple[bool, int]:
        """
        Returns (can_move_this_turn, roll_total).
        Rule: In jail you may roll for doubles up to 3 turns.
        If doubles: leave jail and move that roll.
        If not doubles: increment jail_turns and end turn.
        After 3 failed attempts: pay $50 (if possible), leave jail, move by last roll.
        """
        d1, d2, total, is_double = self.roll_dice()
        self.last_roll = total

        if is_double:
            player.is_in_jail = False
            player.jail_turns = 0
            return True, total

        player.jail_turns += 1

        if player.jail_turns >= 3:
            # Pay $50 to get out after third failed attempt (official)
            player.remove_funds(50)
            player.is_in_jail = False
            player.jail_turns = 0
            return True, total

        return False, total



    def calculate_rent(self, space: Space) -> int:
        # Mortgaged properties collect no rent (official rule)
        if getattr(space, "mortgaged", False):
            return 0

        if space.type == "property":
            # Houses/hotel rent from rent table
            if space.houses > 0:
                # houses 1..4 map to rent[1..4]; hotel (5) maps to rent[5]
                idx = min(space.houses, 5)
                return space.rent[idx]

            # No houses: monopoly doubles base rent
            if space.owner and space.owner.owns_monopoly(space.color, self.board):
                return space.rent[0] * 2

            return space.rent[0]

        if space.type == "railroad":
            owner = space.owner
            if not owner:
                return 0
            rr_owned = sum(1 for p in owner.properties if p.type == "railroad" and not getattr(p, "mortgaged", False))
            # Official RR rent: 25, 50, 100, 200 for 1..4 railroads
            return {1: 25, 2: 50, 3: 100, 4: 200}.get(rr_owned, 0)

        if space.type == "utility":
            owner = space.owner
            if not owner:
                return 0
            util_owned = sum(1 for p in owner.properties if p.type == "utility" and not getattr(p, "mortgaged", False))
            # Official utilities: 4x dice for 1 utility, 10x dice for 2 utilities
            multiplier = 4 if util_owned == 1 else 10 if util_owned == 2 else 0
            return multiplier * self.last_roll

        return 0

    def should_buy_unowned(self, player: Player, space: Space, strategy: str) -> bool:
        """Strategy decision: buy now at face value? (Auction happens if False.)"""
        if space.type not in {"property", "railroad", "utility"}:
            return False
        if space.owner is not None:
            return False
        if player.balance < space.price:
            return False

        if strategy == "Aggressive":
            return True

        if strategy == "Cautious":
            cautious_min = int(self.params.get("cautious_min", 500))
            return (player.balance - space.price) >= cautious_min

        if strategy == "RailRoadTycoon":
            if space.type in {"railroad", "utility"}:
                return True
            return (player.balance - space.price) >= 700

        if strategy == "ColorCollector":
            already_owns_color = space.color is not None and any(p.color == space.color for p in player.properties)
            return already_owns_color or (player.balance - space.price) >= 400

        return False


    def max_bid(self, bidder: Player, space: Space, strategy: str) -> int:
        """How much this bidder is willing to pay in an auction (0 means won't bid)."""
        if bidder.balance <= 0:
            return 0

        # Default: don't bid above your balance
        bal = bidder.balance

        if strategy == "Aggressive":
            # tends to overpay a little
            return min(bal, int(space.price * 1.25))

        if strategy == "Cautious":
            cautious_min = int(self.params.get("cautious_min", 500))
            # keep cash reserve
            return max(0, min(bal - cautious_min, space.price))

        if strategy == "RailRoadTycoon":
            if space.type == "railroad":
                return min(bal, int(space.price * 1.5))
            if space.type == "utility":
                return min(bal, int(space.price * 1.2))
            return min(bal, int(space.price * 0.9))

        if strategy == "ColorCollector":
            # bids more if it matches a color you already own (set-building pressure)
            if space.type == "property" and space.color is not None:
                owns_same_color = any(p.color == space.color for p in bidder.properties)
                return min(bal, int(space.price * (1.4 if owns_same_color else 1.0)))
            return min(bal, space.price)

        return min(bal, space.price)


    def run_auction(self, space: Space) -> None:
        """Official rule: auction unowned properties when a player declines to buy."""
        if space.owner is not None:
            return
        if space.type not in {"property", "railroad", "utility"}:
            return

        # Compute each player's max bid
        bids = []
        for i, p in enumerate(self.players):
            strat = self.strategies[i]
            bids.append((self.max_bid(p, space, strat), i))

        # Highest bid wins (tie-break randomly)
        bids.sort(reverse=True, key=lambda x: x[0])
        top_bid = bids[0][0]
        if top_bid <= 0:
            return  # nobody bids

        tied = [idx for bid, idx in bids if bid == top_bid]
        winner_index = self.rng.choice(tied)
        winner = self.players[winner_index]

        # Pay & transfer
        winner.remove_funds(top_bid)
        winner.properties.append(space)
        space.owner = winner


    def handle_property_logic(self, player: Player, space: Space, strategy: str) -> str:

        # 1) UNOWNED PROPERTY → BUY OR AUCTION
        if space.type in {"property", "railroad", "utility"} and space.owner is None:
            if self.should_buy_unowned(player, space, strategy):
                player.buy_property(space)
            else:
                self.run_auction(space)
            return "OK"

        # 2) OWNED BY SOMEONE ELSE → PAY RENT
        if space.owner and space.owner is not player:
            rent = self.calculate_rent(space)
            success = player.remove_funds(rent)
            space.owner.add_funds(rent)
            return "OK" if success else "BANKRUPT"

        # 3) EVERYTHING ELSE → DO NOTHING
        return "OK"

    def handle_housing_logic(self, player: Player, strategy: str) -> None:
        for prop in player.properties:
            if prop.type != "property" or prop.color is None:
                continue
            if not player.owns_monopoly(prop.color, self.board):
                continue
            if prop.houses >= 5:
                continue

            if strategy == "Aggressive" and player.balance > prop.house_cost:
                player.remove_funds(prop.house_cost)
                prop.houses += 1
            elif strategy == "Cautious" and (player.balance - prop.house_cost) > 1000:
                player.remove_funds(prop.house_cost)
                prop.houses += 1


    def run_turn(self, player_index):
        player = self.players[player_index]
        strategy = self.strategies[player_index]

        if player.balance <= 0:
            return False

        consecutive_doubles = 0
        continue_turn = True

        while continue_turn:
            continue_turn = False  # only becomes True again if we roll doubles (and not jailed)

            # --- JAIL HANDLING ---
            if player.is_in_jail:
                can_move, roll_total = self.take_jail_turn(player)
                if not can_move:
                    return True  # turn ends in jail, not bankrupt
            else:
                d1, d2, roll_total, is_double = self.roll_dice()
                self.last_roll = roll_total

                if is_double:
                    consecutive_doubles += 1
                    if consecutive_doubles >= 3:
                        self.send_to_jail(player)
                        return True  # turn ends immediately
                    continue_turn = True
                else:
                    consecutive_doubles = 0

            # --- MOVE ---
            player.move(roll_total)
            current_space = self.board.get_space(player.position)

            # --- GO TO JAIL SPACE ---
            # official: landing on Go To Jail sends you to jail immediately (no property actions)
            if player.position == self.GO_TO_JAIL_INDEX or current_space.name == "Go To Jail":
                self.send_to_jail(player)
                return True

            # --- PROPERTY / RENT / AUCTION LOGIC ---
            status = self.handle_property_logic(player, current_space, strategy)

            if status == "BANKRUPT":
                return False

            # --- BUILDING PHASE ---
            self.handle_housing_logic(player, strategy)

        return True


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
