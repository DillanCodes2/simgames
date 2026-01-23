# monosim/gui_controller.py
from __future__ import annotations

from .Simulator import Controller
from .Board import Space
from .ui_pygame import PygameUI


class PygameHumanController:
    """
    Human controller that delegates decisions to the Pygame UI.
    """
    def __init__(self, ui: PygameUI, name: str = "Human"):
        self.ui = ui
        self._name = name

    def label(self) -> str:
        return self._name

    def decide_buy(self, sim, player_index: int, space: Space) -> bool:
        p = sim.players[player_index]
        if p.balance < space.price:
            self.ui.push_log(f"Not enough cash to buy {space.name}.")
            return False
        return self.ui.request_yes_no(f"Buy {space.name} for ${space.price}? (bal=${p.balance})")

    def decide_max_bid(self, sim, player_index: int, space: Space) -> int:
        p = sim.players[player_index]
        return self.ui.request_max_bid(
            f"Auction: {space.name}. Choose max bid.",
            max_allowed=p.balance,
        )

    def building_phase(self, sim, player_index: int) -> None:
        # Keep it simple for now (same as terminal behavior later).
        return
