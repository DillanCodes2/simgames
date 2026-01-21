# monosim/Simulator.py
import random
from Board import Board
from Player import Player


class Simulation:
    def __init__(self, strategies):
        """
        strategies: A list of strings defining the playstyle for each of the 4 players.
        Example: ["Aggressive", "Aggressive", "Cautious", "Cautious"]
        """
        self.board = Board()
        self.players = [Player(f"P{i+1} ({strat})")
                        for i, strat in enumerate(strategies)]
        self.strategies = strategies
        self.turn_count = 0
        self.game_log = []

    def roll_dice(self):
        return random.randint(1, 6) + random.randint(1, 6)

    def handle_property_logic(self, player, space, strategy):
        """
        Decides whether to buy or pay rent.
        Strategies: Aggressive, Cautious, RailRoadTycoon, ColorCollector
        """
        if space.type in ["property", "railroad", "utility"] and space.owner is None:

            # 1. Aggressive: Buy everything no matter what
            if strategy == "Aggressive":
                player.buy_property(space)

            # 2. Cautious: Only buy if balance remains above $500
            elif strategy == "Cautious":
                if player.balance - space.price >= 500:
                    player.buy_property(space)

            # 3. RailRoadTycoon: Prioritizes Railroads and Utilities above all else
            elif strategy == "RailRoadTycoon":
                if space.type in ["railroad", "utility"]:
                    player.buy_property(space)
                elif player.balance - space.price >= 700:  # Very conservative on normal land
                    player.buy_property(space)

            # 4. ColorCollector: Focuses on finishing a specific color set
            elif strategy == "ColorCollector":
                # Buy if it's a color the player already owns part of
                already_owns_color = any(
                    p.color == space.color for p in player.properties)
                if already_owns_color or player.balance - space.price >= 400:
                    player.buy_property(space)

        elif space.owner and space.owner != player:
            # Standard Rent Logic
            rent = self.calculate_rent(space)
            success = player.remove_funds(rent)
            space.owner.add_funds(rent)
            return "OK" if success else "BANKRUPT"

        return "OK"

    # monosim/Simulator.py


def calculate_rent(self, space):
    if space.type == "property":
        # If owner has a monopoly but 0 houses, rent doubles
        if space.owner.owns_monopoly(space.color, self.board) and space.houses == 0:
            return space.base_rent * 2
        # If houses exist, rent increases significantly (standard Monopoly math)
        elif space.houses > 0:
            # Simplified exponential growth
            return space.base_rent * (5 ** space.houses)

    # ... (keep railroad/utility logic from previous step)
    return space.base_rent


def handle_housing_logic(self, player, strategy):
    """Called at the end of a player's turn to see if they want to build."""
    for prop in player.properties:
        if player.owns_monopoly(prop.color, self.board) and prop.houses < 5:

            # STRATEGY: Aggressive Builder (Builds if they have the cash)
            if strategy == "Aggressive" and player.balance > prop.house_price:
                player.remove_funds(prop.house_price)
                prop.houses += 1

            # STRATEGY: Cautious Builder (Builds only if balance stays > $1000)
            elif strategy == "Cautious" and (player.balance - prop.house_price) > 1000:
                player.remove_funds(prop.house_price)
                prop.houses += 1


def run_turn(self, player_index):
    player = self.players[player_index]
    strategy = self.strategies[player_index]

    # 1. Move and Interact
    roll = self.roll_dice()
    player.move(roll)
    current_space = self.board.get_space(player.position)
    status = self.handle_property_logic(player, current_space, strategy)

    # 2. Build Houses (New step!)
    if status != "BANKRUPT":
        self.handle_housing_logic(player, strategy)

    return status != "BANKRUPT"


def run_full_game(self, max_turns=1000):
    """Runs the game until one player remains or max_turns reached."""
    for _ in range(max_turns):
        self.turn_count += 1
        for i in range(len(self.players)):
            if self.players[i].balance > 0:
                active = self.run_turn(i)

        # Check for winner (if only one player has money)
        active_players = [p for p in self.players if p.balance > 0]
        if len(active_players) <= 1:
            break

    return self.get_results()


def get_results(self):
    """Returns the final state of the players."""
    return [(p.name, p.balance, len(p.properties)) for p in self.players]


# --- Example Execution Block ---
if __name__ == "__main__":
    # Test 1: 2 Aggressive vs 2 Cautious
    playstyles = ["Aggressive", "Aggressive", "Cautious", "Cautious"]
    sim = Simulation(playstyles)
    results = sim.run_full_game()

    print(f"Game finished in {sim.turn_count} turns.")
    for name, balance, props in results:
        print(f"{name}: Balance ${balance}, Properties Owned: {props}")


def run_batch_test(iterations=1000):
    stats = {"Aggressive": 0, "Cautious": 0,
             "RailRoadTycoon": 0, "ColorCollector": 0}

    for _ in range(iterations):
        # Assign one of each strategy to the 4 players
        playstyles = ["Aggressive", "Cautious",
                      "RailRoadTycoon", "ColorCollector"]
        sim = Simulation(playstyles)
        results = sim.run_full_game()

        # Determine winner (highest balance or last one alive)
        winner_name = max(sim.players, key=lambda p: p.balance).name
        # Extract strategy name from the player name string
        for strat in stats.keys():
            if strat in winner_name:
                stats[strat] += 1

    print("\n--- RESULTS AFTER", iterations, "GAMES ---")
    for strat, wins in stats.items():
        win_percentage = (wins / iterations) * 100
        print(f"{strat}: {wins} wins ({win_percentage:.1f}%)")


if __name__ == "__main__":
    run_batch_test(1000)
