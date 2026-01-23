# monosim/Player.py

class Player:
    """Represents a player in the Monopoly simulation."""

    def __init__(self, name, starting_balance=1500): #defines a player's state at the start of the game
        self.name = name
        self.balance = starting_balance
        self.position = 0  # 0 is the index for 'Go'
        self.properties = []  # List of Space objects owned
        self.is_in_jail = False
        self.jail_turns = 0
        self.get_out_of_jail_free_chance = 0
        self.get_out_of_jail_free_chest = 0


    def move(self, roll_total):
        """
        Updates the player's position based on a dice roll.
        Returns True if the player passed 'Go'.
        """
        new_position = (self.position + roll_total) % 40

        # Check if we passed Go (index wrapped around)
        passed_go = new_position < self.position
        self.position = new_position

        if passed_go:
            self.add_funds(200)

        return passed_go

    def add_funds(self, amount):
        """Adds money to the player's balance."""
        self.balance += amount

    def remove_funds(self, amount):
        """
        Removes money from the player's balance.
        Returns True if successful, False if insufficient cash.
        (Bankruptcy/liquidation is handled by Simulation, not here.)
        """
        if self.balance >= amount:
            self.balance -= amount
            return True
        return False


    def buy_property(self, property_space):
        """Adds a property object to the player's collection and deducts cost."""
        if self.balance >= property_space.price:
            self.remove_funds(property_space.price)
            self.properties.append(property_space)
            property_space.owner = self  # Link the space back to this player
            return True
        return False

    def get_owned_color_count(self, color): #Returns the current number of owned colors
        return sum(1 for p in self.properties if p.color == color)

    def owns_monopoly(self, color, board): #Returns whether or not a player owns all properties of one color
        if color is None:
            return False
        return self.get_owned_color_count(color) == board.get_color_count(color)

    def __repr__(self):
        """Technical string representation for debugging."""
        return f"Player({self.name}, Balance: ${self.balance}, Pos: {self.position})"
    
    def has_get_out_of_jail_free(self) -> bool:
        return (self.get_out_of_jail_free_chance + self.get_out_of_jail_free_chest) > 0

    def use_get_out_of_jail_free(self) -> str | None:
        """Returns which deck the used card came from: 'chance' or 'chest'."""
        if self.get_out_of_jail_free_chance > 0:
            self.get_out_of_jail_free_chance -= 1
            return "chance"
        if self.get_out_of_jail_free_chest > 0:
            self.get_out_of_jail_free_chest -= 1
            return "chest"
        return None

