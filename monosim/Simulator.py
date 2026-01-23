# monosim/Simulator.py
from __future__ import annotations

import random
from collections import Counter
from typing import List, Dict, Tuple, Optional

from .Board import Board, Space
from .Player import Player

KNOWN_STRATEGIES = {"Aggressive", "Cautious", "RailRoadTycoon", "ColorCollector"}


# Canonical ordering helpers (seat order is ignored everywhere in this project).
STRATEGY_ORDER = tuple(sorted(KNOWN_STRATEGIES))


def lineup_tag(strategies: List[str]) -> str:
    """
    Stable, human-readable identifier for a 4-player lineup, ignoring seating.
    Example: "Aggressive=2|Cautious=1|RailRoadTycoon=1"
    """
    counts = {s: 0 for s in STRATEGY_ORDER}
    for s in strategies:
        if s not in KNOWN_STRATEGIES:
            raise ValueError(f"Unknown strategy '{s}'")
        counts[s] += 1
    parts = [f"{s}={counts[s]}" for s in STRATEGY_ORDER if counts[s] > 0]
    return "|".join(parts) if parts else "EMPTY"


def generate_all_lineups(total_players: int = 4) -> List[List[str]]:
    """
    Generate every unique playstyle permutation where only counts matter (not seating).
    With 4 strategies and 4 players, there are C(7,4)=35 lineups.
    """
    if total_players <= 0:
        return []

    strategies = list(STRATEGY_ORDER)

    results: List[List[str]] = []

    def rec(i: int, remaining: int, acc_counts: List[int]) -> None:
        if i == len(strategies) - 1:
            acc = acc_counts + [remaining]
            lineup: List[str] = []
            for strat, c in zip(strategies, acc):
                lineup.extend([strat] * c)
            results.append(lineup)
            return

        for c in range(remaining + 1):
            rec(i + 1, remaining - c, acc_counts + [c])

    rec(0, total_players, [])
    return results


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
        self._init_decks()


    def _init_decks(self) -> None:
        # Each deck is a list of (card_name, handler_function, keepable)
        self.chance_deck = self._build_chance_deck()
        self.chest_deck = self._build_chest_deck()
        self.rng.shuffle(self.chance_deck)
        self.rng.shuffle(self.chest_deck)

    def roll_dice(self):
        d1 = self.rng.randint(1, 6)
        d2 = self.rng.randint(1, 6)
        return d1, d2, d1 + d2, (d1 == d2)

    JAIL_INDEX = 10        # "Just Visiting / In Jail" is index 10 on standard board
    GO_TO_JAIL_INDEX = 30  # "Go To Jail" is index 30
    # Standard board indices (matching your Board ordering)
    GO_INDEX = 0
    JAIL_INDEX = 10
    GO_TO_JAIL_INDEX = 30

    READING_RR_INDEX = 5
    ST_CHARLES_INDEX = 11
    ILLINOIS_INDEX = 24
    BOARDWALK_INDEX = 39

    CHANCE_INDICES = {7, 22, 36}
    CHEST_INDICES = {2, 17, 33}

    def can_mortgage(self, prop: Space, owner: Player) -> bool:
        if prop.owner is not owner:
            return False
        if prop.mortgaged:
            return False
        if prop.type == "property":
            # must have no buildings on this property
            if prop.houses != 0:
                return False
            # (baseline rule) also require no buildings on any property of this color owned by player
            if prop.color is not None:
                for s in owner.properties:
                    if s.type == "property" and s.color == prop.color and s.houses != 0:
                        return False
        return prop.type in {"property", "railroad", "utility"}


    def sell_one_building(self, owner: Player) -> bool:
        """
        Sell ONE house/hotel (half cost). Returns True if sold anything.
        We use a simple heuristic: sell the most expensive building first.
        """
        buildables = [p for p in owner.properties if p.type == "property" and p.houses > 0]
        if not buildables:
            return False

        # choose the property with the highest house_cost to liquidate first
        prop = max(buildables, key=lambda p: p.house_cost)

        # Treat hotel as 5 "building units" for liquidation value; otherwise sell 1 house.
        if prop.houses == 5:
            sale_value = int((prop.house_cost * 5) / 2)  # half of 5 houses
            prop.houses = 4  # (baseline) downgrade hotel to 4 houses; we aren't modeling house supply limits yet
        else:
            sale_value = int(prop.house_cost / 2)
            prop.houses -= 1

        owner.add_funds(sale_value)
        return True


    def mortgage_one_property(self, owner: Player) -> bool:
        """
        Mortgage ONE property (collect mortgage value).
        Returns True if mortgaged something.
        """
        candidates = [p for p in owner.properties if self.can_mortgage(p, owner)]
        if not candidates:
            return False

        # mortgage lowest-rent-impact first
        prop = min(candidates, key=lambda p: p.mortgage)
        prop.mortgaged = True
        owner.add_funds(prop.mortgage)
        return True


    def liquidate_until_can_pay(self, owner: Player, amount: int) -> bool:
        """
        Try to raise cash via selling buildings then mortgaging.
        Return True if owner can pay amount after liquidation.
        """
        safety = 0
        while owner.balance < amount and safety < 500:
            safety += 1
            if self.sell_one_building(owner):
                continue
            if self.mortgage_one_property(owner):
                continue
            break
        return owner.balance >= amount


    def transfer_assets_to_player(self, bankrupt: Player, creditor: Player) -> None:
        """
        Transfer all properties to creditor, preserving mortgaged state.
        """
        for prop in list(bankrupt.properties):
            prop.owner = creditor
            creditor.properties.append(prop)
        bankrupt.properties.clear()


    def return_assets_to_bank(self, bankrupt: Player) -> None:
        """Return properties to bank (unowned, unmortgaged, no buildings)."""
        for prop in list(bankrupt.properties):
            prop.owner = None
            prop.mortgaged = False
            prop.houses = 0
        bankrupt.properties.clear()


    def pay(self, payer: Player, amount: int, creditor: Player | None = None) -> bool:
        """
        Core payment routine implementing liquidation + bankruptcy.
        creditor=None means bank.
        Returns True if paid, False if payer bankrupt.
        """
        if amount <= 0:
            return True

        # Try to raise money before failing
        if payer.balance < amount:
            self.liquidate_until_can_pay(payer, amount)

        if payer.balance >= amount:
            payer.balance -= amount
            if creditor is not None:
                creditor.add_funds(amount)
            return True

        # Still can't pay -> bankruptcy
        payer.balance = 0

        # Before transferring assets, sell off remaining buildings (officially must)
        while self.sell_one_building(payer):
            pass

        if creditor is not None:
            self.transfer_assets_to_player(payer, creditor)
        else:
            self.return_assets_to_bank(payer)

        return False


    def resolve_landing(self, player_index: int) -> str:
        """Resolve the current space: cards, taxes, jail, property, etc."""
        player = self.players[player_index]
        strategy = self.strategies[player_index]

        safety = 0
        while safety < 10:
            safety += 1
            space = self.board.get_space(player.position)

            # Go To Jail
            if player.position == self.GO_TO_JAIL_INDEX or space.name == "Go To Jail":
                self.send_to_jail(player)
                return "OK"

            # Chance / Community Chest
            if space.name == "Chance" or player.position in self.CHANCE_INDICES:
                self.draw_card("chance", player_index)
                continue

            if space.name == "Community Chest" or player.position in self.CHEST_INDICES:
                self.draw_card("chest", player_index)
                continue

            # Taxes
            if space.name == "Income Tax":
                ok = self.pay(player, 200, creditor=None)
                return "OK" if ok else "BANKRUPT"

            if space.name == "Luxury Tax":
                ok = self.pay(player, 100, creditor=None)
                return "OK" if ok else "BANKRUPT"

            status = self.handle_property_logic(player, space, strategy)
            return status

        return "OK"


    def _build_chest_deck(self):
        def advance_to_go(pi):
            self.move_to(self.players[pi], self.GO_INDEX, collect_go=True)
            return False

        def bank_error_200(pi):
            self.players[pi].add_funds(200)
            return False

        def doctors_fee_50(pi):
            self.pay(self.players[pi], 50, creditor=None)
            return False

        def sale_of_stock_50(pi):
            self.players[pi].add_funds(50)
            return False

        def get_out_of_jail_free(pi):
            self.players[pi].get_out_of_jail_free_chest += 1
            return True

        def go_to_jail(pi):
            self.send_to_jail(self.players[pi])
            return False

        def holiday_fund_100(pi):
            self.players[pi].add_funds(100)
            return False

        def income_tax_refund_20(pi):
            self.players[pi].add_funds(20)
            return False

        def birthday_collect_10_each(pi):
            receiver = self.players[pi]
            for j, other in enumerate(self.players):
                if j == pi or other.balance <= 0:
                    continue
                self.pay(other, 10, creditor=receiver)
            return False

        def life_insurance_100(pi):
            self.players[pi].add_funds(100)
            return False

        def hospital_fee_100(pi):
            self.pay(self.players[pi], 100, creditor=None)
            return False

        def school_fee_150(pi):
            self.pay(self.players[pi], 150, creditor=None)
            return False

        def consultancy_25(pi):
            self.players[pi].add_funds(25)
            return False

        def street_repairs(pi):
            player = self.players[pi]
            houses, hotels = self.house_hotel_counts(player)
            cost = houses * 40 + hotels * 115
            self.pay(player, cost, creditor=None)
            return False

        def beauty_prize_10(pi):
            self.players[pi].add_funds(10)
            return False

        def inherit_100(pi):
            self.players[pi].add_funds(100)
            return False

        return [
            ("Advance to Go", advance_to_go, False),
            ("Bank error in your favor ($200)", bank_error_200, False),
            ("Doctor's fee ($50)", doctors_fee_50, False),
            ("From sale of stock you get $50", sale_of_stock_50, False),
            ("Get Out of Jail Free", get_out_of_jail_free, True),
            ("Go to Jail", go_to_jail, False),
            ("Holiday fund matures ($100)", holiday_fund_100, False),
            ("Income tax refund ($20)", income_tax_refund_20, False),
            ("It is your birthday — collect $10 from each", birthday_collect_10_each, False),
            ("Life insurance matures ($100)", life_insurance_100, False),
            ("Pay hospital fees ($100)", hospital_fee_100, False),
            ("Pay school fees ($150)", school_fee_150, False),
            ("Receive for services ($25)", consultancy_25, False),
            ("You are assessed for street repairs", street_repairs, False),
            ("Second prize in a beauty contest ($10)", beauty_prize_10, False),
            ("You inherit $100", inherit_100, False),
        ]


    def _build_chance_deck(self):
        def advance_to_go(pi): 
            self.move_to(self.players[pi], self.GO_INDEX, collect_go=True)
            return False

        def advance_to_illinois(pi):
            self.move_to(self.players[pi], self.ILLINOIS_INDEX, collect_go=True)
            return False

        def advance_to_st_charles(pi):
            self.move_to(self.players[pi], self.ST_CHARLES_INDEX, collect_go=True)
            return False

        def advance_to_boardwalk(pi):
            self.move_to(self.players[pi], self.BOARDWALK_INDEX, collect_go=True)
            return False

        def trip_to_reading(pi):
            self.move_to(self.players[pi], self.READING_RR_INDEX, collect_go=True)
            return False

        def nearest_rr_pay_double(pi):
            player = self.players[pi]
            dest = self.nearest_railroad(player.position)
            self.move_to(player, dest, collect_go=True)
            rr = self.board.get_space(dest)
            if rr.owner is None or rr.owner is player:
                return False
            rent = self.calculate_rent(rr) * 2
            self.pay(player, rent, creditor=rr.owner)
            return False

        def nearest_util(pi):
            player = self.players[pi]
            dest = self.nearest_utility(player.position)
            self.move_to(player, dest, collect_go=True)
            u = self.board.get_space(dest)
            if u.owner is None or u.owner is player:
                return False
            d1, d2, total, _ = self.roll_dice()
            self.last_roll = total
            rent = 10 * total
            self.pay(player, rent, creditor=u.owner)
            return False

        def bank_dividend_50(pi):
            self.players[pi].add_funds(50)
            return False

        def poor_tax_15(pi):
            self.pay(self.players[pi], 15, creditor=None)
            return False

        def building_loan_150(pi):
            self.players[pi].add_funds(150)
            return False

        def chairman_pay_50_each(pi):
            payer = self.players[pi]
            for j, other in enumerate(self.players):
                if j == pi or other.balance <= 0:
                    continue
                self.pay(payer, 50, creditor=other)
            return False

        def general_repairs(pi):
            player = self.players[pi]
            houses, hotels = self.house_hotel_counts(player)
            cost = houses * 25 + hotels * 100
            self.pay(player, cost, creditor=None)
            return False

        def go_back_3(pi):
            player = self.players[pi]
            player.position = (player.position - 3) % 40
            return False

        def go_to_jail(pi):
            self.send_to_jail(self.players[pi])
            return False

        def get_out_of_jail_free(pi):
            self.players[pi].get_out_of_jail_free_chance += 1
            return True

        return [
            ("Advance to Go", advance_to_go, False),
            ("Advance to Illinois Avenue", advance_to_illinois, False),
            ("Advance to St. Charles Place", advance_to_st_charles, False),
            ("Advance to Boardwalk", advance_to_boardwalk, False),
            ("Take a trip to Reading Railroad", trip_to_reading, False),
            ("Advance to nearest Railroad (pay double)", nearest_rr_pay_double, False),
            ("Advance to nearest Railroad (pay double)", nearest_rr_pay_double, False),
            ("Advance to nearest Utility", nearest_util, False),
            ("Bank pays you dividend of $50", bank_dividend_50, False),
            ("Pay poor tax of $15", poor_tax_15, False),
            ("Your building and loan matures ($150)", building_loan_150, False),
            ("Elected Chairman — pay each player $50", chairman_pay_50_each, False),
            ("Make general repairs", general_repairs, False),
            ("Go back 3 spaces", go_back_3, False),
            ("Go to Jail", go_to_jail, False),
            ("Get Out of Jail Free", get_out_of_jail_free, True),
        ]


    def draw_card(self, deck_name: str, player_index: int) -> None:
        deck = self.chance_deck if deck_name == "chance" else self.chest_deck
        card_name, fn, keepable = deck.pop(0)

        keep = fn(player_index)
        if keepable and keep:
            return

        deck.append((card_name, fn, keepable))


    def return_get_out_of_jail_free_to_bottom(self, deck_name: str) -> None:
        if deck_name == "chance":
            for i, (name, fn, keepable) in enumerate(self._build_chance_deck()):
                if name == "Get Out of Jail Free":
                    self.chance_deck.append((name, fn, keepable))
                    return
        else:
            for i, (name, fn, keepable) in enumerate(self._build_chest_deck()):
                if name == "Get Out of Jail Free":
                    self.chest_deck.append((name, fn, keepable))
                    return


    def move_to(self, player: Player, index: int, collect_go: bool = True) -> None:
        index %= 40
        if collect_go and index < player.position:
            player.add_funds(200)
        player.position = index

    def nearest_index_of_type(self, start_pos: int, space_type: str) -> int:
        for step in range(1, 41):
            idx = (start_pos + step) % 40
            if self.board.get_space(idx).type == space_type:
                return idx
        return start_pos

    def nearest_railroad(self, start_pos: int) -> int:
        return self.nearest_index_of_type(start_pos, "railroad")

    def nearest_utility(self, start_pos: int) -> int:
        return self.nearest_index_of_type(start_pos, "utility")

    def house_hotel_counts(self, player: Player) -> tuple[int, int]:
        houses = 0
        hotels = 0
        for p in player.properties:
            if p.type == "property":
                if p.houses == 5:
                    hotels += 1
                elif p.houses > 0:
                    houses += p.houses
        return houses, hotels



    def send_to_jail(self, player: Player):
        player.position = self.JAIL_INDEX
        player.is_in_jail = True
        player.jail_turns = 0

    def take_jail_turn(self, player: Player) -> tuple[bool, int]:
        d1, d2, total, is_double = self.roll_dice()
        self.last_roll = total

        if is_double:
            player.is_in_jail = False
            player.jail_turns = 0
            return True, total

        player.jail_turns += 1

        if player.jail_turns >= 3:
            ok = self.pay(player, 50, creditor=None)
            player.is_in_jail = False
            player.jail_turns = 0
            return True, total

        return False, total



    def calculate_rent(self, space: Space) -> int:
        if getattr(space, "mortgaged", False):
            return 0

        if space.type == "property":
            if space.houses > 0:
                idx = min(space.houses, 5)
                return space.rent[idx]

            if space.owner and space.owner.owns_monopoly(space.color, self.board):
                return space.rent[0] * 2

            return space.rent[0]

        if space.type == "railroad":
            owner = space.owner
            if not owner:
                return 0
            rr_owned = sum(1 for p in owner.properties if p.type == "railroad" and not getattr(p, "mortgaged", False))
            return {1: 25, 2: 50, 3: 100, 4: 200}.get(rr_owned, 0)

        if space.type == "utility":
            owner = space.owner
            if not owner:
                return 0
            util_owned = sum(1 for p in owner.properties if p.type == "utility" and not getattr(p, "mortgaged", False))
            multiplier = 4 if util_owned == 1 else 10 if util_owned == 2 else 0
            return multiplier * self.last_roll

        return 0

    # (rest of your file unchanged)
    def should_buy_unowned(self, player: Player, space: Space, strategy: str) -> bool:
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
        if bidder.balance <= 0:
            return 0

        bal = bidder.balance

        if strategy == "Aggressive":
            return min(bal, int(space.price * 1.25))

        if strategy == "Cautious":
            cautious_min = int(self.params.get("cautious_min", 500))
            return max(0, min(bal - cautious_min, space.price))

        if strategy == "RailRoadTycoon":
            if space.type == "railroad":
                return min(bal, int(space.price * 1.5))
            if space.type == "utility":
                return min(bal, int(space.price * 1.2))
            return min(bal, int(space.price * 0.9))

        if strategy == "ColorCollector":
            if space.type == "property" and space.color is not None:
                owns_same_color = any(p.color == space.color for p in bidder.properties)
                return min(bal, int(space.price * (1.4 if owns_same_color else 1.0)))
            return min(bal, space.price)

        return min(bal, space.price)


    def run_auction(self, space: Space) -> None:
        if space.owner is not None:
            return
        if space.type not in {"property", "railroad", "utility"}:
            return

        bids = []
        for i, p in enumerate(self.players):
            strat = self.strategies[i]
            bids.append((self.max_bid(p, space, strat), i))

        bids.sort(reverse=True, key=lambda x: x[0])
        top_bid = bids[0][0]
        if top_bid <= 0:
            return

        tied = [idx for bid, idx in bids if bid == top_bid]
        winner_index = self.rng.choice(tied)
        winner = self.players[winner_index]

        ok = self.pay(winner, top_bid, creditor=None)
        if not ok:
            return

        winner.properties.append(space)
        space.owner = winner


    def handle_property_logic(self, player: Player, space: Space, strategy: str) -> str:
        if space.type in {"property", "railroad", "utility"} and space.owner is None:
            if self.should_buy_unowned(player, space, strategy):
                player.buy_property(space)
            else:
                self.run_auction(space)
            return "OK"

        if space.owner and space.owner is not player:
            rent = self.calculate_rent(space)
            ok = self.pay(player, rent, creditor=space.owner)
            return "OK" if ok else "BANKRUPT"

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
                ok = self.pay(player, prop.house_cost, creditor=None)
                if ok:
                    prop.houses += 1
            elif strategy == "Cautious" and (player.balance - prop.house_cost) > 1000:
                ok = self.pay(player, prop.house_cost, creditor=None)
                if ok:
                    prop.houses += 1

    # ... rest of file unchanged (run_turn, run_full_game, run_batch)



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
            if player.is_in_jail and player.has_get_out_of_jail_free():
                used_from = player.use_get_out_of_jail_free()
                player.is_in_jail = False
                player.jail_turns = 0
                if used_from:
                    self.return_get_out_of_jail_free_to_bottom(used_from)

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
            status = self.resolve_landing(player_index)

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
