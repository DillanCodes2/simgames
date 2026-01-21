# monosim/Board.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Space:
    name: str
    type: str
    price: int = 0
    base_rent: int = 0
    color: Optional[str] = None

    houses: int = 0         # 0..5 (5 = hotel)
    house_price: int = 50   # simplified: constant build cost
    owner: Optional["Player"] = None  # set when bought


class Board:
    """Manages the collection of all spaces on the board."""

    def __init__(self):
        self.spaces: List[Space] = self._initialize_board()

    def _initialize_board(self) -> List[Space]:
        # Standard Monopoly board (cards/taxes/etc are placeholders).
        return [
            Space("Go", "special"),
            Space("Mediterranean Avenue", "property", 60, 2, "brown"),
            Space("Community Chest", "special"),
            Space("Baltic Avenue", "property", 60, 4, "brown"),
            Space("Income Tax", "special"),
            Space("Reading Railroad", "railroad", 200, 25),
            Space("Oriental Avenue", "property", 100, 6, "light_blue"),
            Space("Chance", "special"),
            Space("Vermont Avenue", "property", 100, 6, "light_blue"),
            Space("Connecticut Avenue", "property", 120, 8, "light_blue"),
            Space("Just Visiting", "special"),
            Space("St. Charles Place", "property", 140, 10, "pink"),
            Space("Electric Company", "utility", 150, 0),
            Space("States Avenue", "property", 140, 10, "pink"),
            Space("Virginia Avenue", "property", 160, 12, "pink"),
            Space("Pennsylvania Railroad", "railroad", 200, 25),
            Space("St. James Place", "property", 180, 14, "orange"),
            Space("Community Chest", "special"),
            Space("Tennessee Avenue", "property", 180, 14, "orange"),
            Space("New York Avenue", "property", 200, 16, "orange"),
            Space("Free Parking", "special"),
            Space("Kentucky Avenue", "property", 220, 18, "red"),
            Space("Chance", "special"),
            Space("Indiana Avenue", "property", 220, 18, "red"),
            Space("Illinois Avenue", "property", 240, 20, "red"),
            Space("B. & O. Railroad", "railroad", 200, 25),
            Space("Atlantic Avenue", "property", 260, 22, "yellow"),
            Space("Ventnor Avenue", "property", 260, 22, "yellow"),
            Space("Water Works", "utility", 150, 0),
            Space("Marvin Gardens", "property", 280, 24, "yellow"),
            Space("Go To Jail", "special"),
            Space("Pacific Avenue", "property", 300, 26, "green"),
            Space("North Carolina Avenue", "property", 300, 26, "green"),
            Space("Community Chest", "special"),
            Space("Pennsylvania Avenue", "property", 320, 28, "green"),
            Space("Short Line", "railroad", 200, 25),
            Space("Chance", "special"),
            Space("Park Place", "property", 350, 35, "dark_blue"),
            Space("Luxury Tax", "special"),
            Space("Boardwalk", "property", 400, 50, "dark_blue"),
        ]

    def get_space(self, index: int) -> Space:
        return self.spaces[index % 40]

    def get_color_count(self, color: str) -> int:
        """How many properties of a given color exist on the board."""
        return sum(1 for s in self.spaces if s.color == color)

    def get_property_details(self, index: int) -> dict:
        """Helper to get specific info for the simulator/debugging."""
        s = self.get_space(index)
        return {
            "name": s.name,
            "cost": s.price,
            "base_rent": s.base_rent,
            "type": s.type,
            "color": s.color,
            "owner": None if s.owner is None else s.owner.name,
            "houses": s.houses,
        }
