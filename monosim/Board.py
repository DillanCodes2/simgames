# monosim/Board.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .Player import Player



@dataclass
class Space:
    name: str
    type: str  # "property", "railroad", "utility", "special"
    price: int = 0
    color: Optional[str] = None

    # For standard color properties: rent[0]=no houses, rent[1..4]=1-4 houses, rent[5]=hotel
    rent: List[int] = field(default_factory=list)

    # House cost (varies by color set in official rules)
    house_cost: int = 0

    # Mortgage value (official: typically price/2 for properties; railroads/utilities also have mortgage values)
    mortgage: int = 0
    mortgaged: bool = False

    # Buildings: 0..4 houses, 5 = hotel
    houses: int = 0

    owner: Optional["Player"] = None


class Board:
    def __init__(self):
        self.spaces: List[Space] = self._initialize_board()

    def _initialize_board(self) -> List[Space]:
        # Official rent tables (US Monopoly). These are the standard values.
        # Format: [rent, 1h, 2h, 3h, 4h, hotel]
        def prop(name, price, color, rent, house_cost):
            return Space(
                name=name,
                type="property",
                price=price,
                color=color,
                rent=rent,
                house_cost=house_cost,
                mortgage=price // 2,
            )

        def rr(name):
            return Space(
                name=name,
                type="railroad",
                price=200,
                mortgage=100,
            )

        def util(name):
            return Space(
                name=name,
                type="utility",
                price=150,
                mortgage=75,
            )

        return [
            Space("Go", "special"),

            prop("Mediterranean Avenue", 60, "brown", [2, 10, 30, 90, 160, 250], 50),
            Space("Community Chest", "special"),
            prop("Baltic Avenue", 60, "brown", [4, 20, 60, 180, 320, 450], 50),
            Space("Income Tax", "special"),

            rr("Reading Railroad"),

            prop("Oriental Avenue", 100, "light_blue", [6, 30, 90, 270, 400, 550], 50),
            Space("Chance", "special"),
            prop("Vermont Avenue", 100, "light_blue", [6, 30, 90, 270, 400, 550], 50),
            prop("Connecticut Avenue", 120, "light_blue", [8, 40, 100, 300, 450, 600], 50),

            Space("Just Visiting", "special"),

            prop("St. Charles Place", 140, "pink", [10, 50, 150, 450, 625, 750], 100),
            util("Electric Company"),
            prop("States Avenue", 140, "pink", [10, 50, 150, 450, 625, 750], 100),
            prop("Virginia Avenue", 160, "pink", [12, 60, 180, 500, 700, 900], 100),

            rr("Pennsylvania Railroad"),

            prop("St. James Place", 180, "orange", [14, 70, 200, 550, 750, 950], 100),
            Space("Community Chest", "special"),
            prop("Tennessee Avenue", 180, "orange", [14, 70, 200, 550, 750, 950], 100),
            prop("New York Avenue", 200, "orange", [16, 80, 220, 600, 800, 1000], 100),

            Space("Free Parking", "special"),

            prop("Kentucky Avenue", 220, "red", [18, 90, 250, 700, 875, 1050], 150),
            Space("Chance", "special"),
            prop("Indiana Avenue", 220, "red", [18, 90, 250, 700, 875, 1050], 150),
            prop("Illinois Avenue", 240, "red", [20, 100, 300, 750, 925, 1100], 150),

            rr("B. & O. Railroad"),

            prop("Atlantic Avenue", 260, "yellow", [22, 110, 330, 800, 975, 1150], 150),
            prop("Ventnor Avenue", 260, "yellow", [22, 110, 330, 800, 975, 1150], 150),
            util("Water Works"),
            prop("Marvin Gardens", 280, "yellow", [24, 120, 360, 850, 1025, 1200], 150),

            Space("Go To Jail", "special"),

            prop("Pacific Avenue", 300, "green", [26, 130, 390, 900, 1100, 1275], 200),
            prop("North Carolina Avenue", 300, "green", [26, 130, 390, 900, 1100, 1275], 200),
            Space("Community Chest", "special"),
            prop("Pennsylvania Avenue", 320, "green", [28, 150, 450, 1000, 1200, 1400], 200),

            rr("Short Line"),

            Space("Chance", "special"),

            prop("Park Place", 350, "dark_blue", [35, 175, 500, 1100, 1300, 1500], 200),
            Space("Luxury Tax", "special"),
            prop("Boardwalk", 400, "dark_blue", [50, 200, 600, 1400, 1700, 2000], 200),
        ]

    def get_space(self, index: int) -> Space:
        return self.spaces[index % 40]

    def get_color_count(self, color: str) -> int:
        return sum(1 for s in self.spaces if s.type == "property" and s.color == color)

    def get_property_details(self, index: int) -> dict:
        s = self.get_space(index)
        return {
            "name": s.name,
            "cost": s.price,
            "type": s.type,
            "color": s.color,
            "owner": None if s.owner is None else s.owner.name,
            "houses": s.houses,
            "mortgaged": s.mortgaged,
            "rent": s.rent,
            "house_cost": s.house_cost,
            "mortgage": s.mortgage,
        }
