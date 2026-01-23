# monosim/ui_pygame.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, List

import pygame

from .Simulator import Simulation
from .Board import Space

WINDOW_W, WINDOW_H = 1200, 800
BOARD_SIZE = 700
MARGIN = 30
PANEL_W = WINDOW_W - (BOARD_SIZE + 2 * MARGIN)

FPS = 60


@dataclass
class UIButton:
    rect: pygame.Rect
    text: str
    on_click: Callable[[], None]
    enabled: bool = True


class PygameUI:
    """
    Minimal GUI shell:
    - draws board + tokens
    - shows status + log
    - provides a simple button system for decisions
    """

    def __init__(self, sim: Simulation):
        self.sim = sim

        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None
        self.small: Optional[pygame.font.Font] = None

        self.log_lines: List[str] = []
        self.buttons: List[UIButton] = []

        # Human decision “mailbox”
        self._pending_yesno: Optional[bool] = None
        self._pending_bid: Optional[int] = None

        # Simple state
        self.running = True
        self.clock = pygame.time.Clock()

    # ---------- Logging ----------
    def push_log(self, msg: str) -> None:
        self.log_lines.append(msg)
        self.log_lines = self.log_lines[-18:]  # keep last N

    # ---------- Decisions ----------
    def request_yes_no(self, prompt: str) -> bool:
        self._pending_yesno = None
        self.push_log(prompt)

        def yes(): self._set_yesno(True)
        def no(): self._set_yesno(False)

        self.buttons = [
            UIButton(pygame.Rect(self._panel_x()+20, 580, 120, 40), "YES", yes),
            UIButton(pygame.Rect(self._panel_x()+160, 580, 120, 40), "NO", no),
        ]

        while self.running and self._pending_yesno is None:
            self._tick()

        self.buttons = []
        return bool(self._pending_yesno)

    def _set_yesno(self, v: bool) -> None:
        self._pending_yesno = v

    def request_max_bid(self, prompt: str, max_allowed: int) -> int:
        """
        Minimal version: uses number keys to pick a bid quickly.
        Better later: a text input widget.
        """
        self._pending_bid = None
        self.push_log(prompt)
        self.push_log("Press 0 to pass. Press 1-9 for 100..900. Press B for full balance.")

        def set0(): self._pending_bid = 0
        def setB(): self._pending_bid = max_allowed

        self.buttons = [
            UIButton(pygame.Rect(self._panel_x()+20, 580, 140, 40), "PASS (0)", set0),
            UIButton(pygame.Rect(self._panel_x()+180, 580, 180, 40), f"BID ALL (${max_allowed})", setB),
        ]

        while self.running and self._pending_bid is None:
            self._tick()

        self.buttons = []
        return int(self._pending_bid or 0)

    # ---------- Main loop helpers ----------
    def start(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Monopoly (Phase 2)")
        self.font = pygame.font.SysFont(None, 28)
        self.small = pygame.font.SysFont(None, 20)

    def _panel_x(self) -> int:
        return MARGIN + BOARD_SIZE + 20

    def _tick(self) -> None:
        assert self.screen and self.font and self.small

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                for b in self.buttons:
                    if b.enabled and b.rect.collidepoint(mx, my):
                        b.on_click()

            if event.type == pygame.KEYDOWN and self._pending_bid is None and self.buttons:
                # bid shortcuts
                if event.key == pygame.K_0:
                    self._pending_bid = 0
                if pygame.K_1 <= event.key <= pygame.K_9:
                    digit = event.key - pygame.K_0
                    self._pending_bid = min(digit * 100, self.sim.players[self.sim.human_index].balance)
                if event.key == pygame.K_b:
                    self._pending_bid = self.sim.players[self.sim.human_index].balance

        self._draw()
        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw(self) -> None:
        assert self.screen and self.font and self.small
        self.screen.fill((25, 25, 28))

        # Board area background
        board_rect = pygame.Rect(MARGIN, MARGIN, BOARD_SIZE, BOARD_SIZE)
        pygame.draw.rect(self.screen, (220, 220, 220), board_rect, border_radius=8)

        self._draw_board(board_rect)
        self._draw_tokens(board_rect)
        self._draw_panel()

    def _draw_board(self, board_rect: pygame.Rect) -> None:
        assert self.screen and self.small

        # Very simple 40-space layout: 10 per side
        spaces = self.sim.board.spaces
        # We just draw space boxes; later we’ll add property colors and names.

        def draw_space(idx: int, rect: pygame.Rect):
            pygame.draw.rect(self.screen, (245, 245, 245), rect)
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
            name = spaces[idx].name
            label = name if len(name) <= 14 else name[:14] + "…"
            txt = self.small.render(label, True, (0, 0, 0))
            self.screen.blit(txt, (rect.x + 4, rect.y + 4))

        cell = BOARD_SIZE // 11  # leaves room for corners
        x0, y0 = board_rect.x, board_rect.y

        # corners: 0,10,20,30
        draw_space(0, pygame.Rect(x0 + 10*cell, y0 + 10*cell, cell, cell))     # Go
        draw_space(10, pygame.Rect(x0, y0 + 10*cell, cell, cell))              # Jail
        draw_space(20, pygame.Rect(x0, y0, cell, cell))                        # Free Parking
        draw_space(30, pygame.Rect(x0 + 10*cell, y0, cell, cell))              # Go To Jail

        # bottom row (1..9) right->left
        for i in range(1, 10):
            idx = i
            rect = pygame.Rect(x0 + (10-i)*cell, y0 + 10*cell, cell, cell)
            draw_space(idx, rect)

        # left column (11..19) bottom->top
        for i in range(1, 10):
            idx = 10 + i
            rect = pygame.Rect(x0, y0 + (10-i)*cell, cell, cell)
            draw_space(idx, rect)

        # top row (21..29) left->right
        for i in range(1, 10):
            idx = 20 + i
            rect = pygame.Rect(x0 + i*cell, y0, cell, cell)
            draw_space(idx, rect)

        # right column (31..39) top->bottom
        for i in range(1, 10):
            idx = 30 + i
            rect = pygame.Rect(x0 + 10*cell, y0 + i*cell, cell, cell)
            draw_space(idx, rect)

    def _draw_tokens(self, board_rect: pygame.Rect) -> None:
        assert self.screen

        # Tiny colored circles; later we’ll make this nicer.
        token_colors = [(220, 50, 50), (50, 220, 50), (50, 140, 220), (220, 220, 50)]
        cell = BOARD_SIZE // 11
        x0, y0 = board_rect.x, board_rect.y

        def space_center(pos: int) -> tuple[int, int]:
            # Map position to the rectangles used above (same logic)
            if pos == 0:   return (x0 + 10*cell + cell//2, y0 + 10*cell + cell//2)
            if pos == 10:  return (x0 + cell//2, y0 + 10*cell + cell//2)
            if pos == 20:  return (x0 + cell//2, y0 + cell//2)
            if pos == 30:  return (x0 + 10*cell + cell//2, y0 + cell//2)

            if 1 <= pos <= 9:
                return (x0 + (10-pos)*cell + cell//2, y0 + 10*cell + cell//2)
            if 11 <= pos <= 19:
                i = pos - 10
                return (x0 + cell//2, y0 + (10-i)*cell + cell//2)
            if 21 <= pos <= 29:
                i = pos - 20
                return (x0 + i*cell + cell//2, y0 + cell//2)
            if 31 <= pos <= 39:
                i = pos - 30
                return (x0 + 10*cell + cell//2, y0 + i*cell + cell//2)

            return (x0 + 10*cell + cell//2, y0 + 10*cell + cell//2)

        for i, p in enumerate(self.sim.players):
            cx, cy = space_center(p.position)
            # offset slightly so tokens don’t stack perfectly
            ox = (i % 2) * 10 - 5
            oy = (i // 2) * 10 - 5
            pygame.draw.circle(self.screen, token_colors[i], (cx + ox, cy + oy), 10)

    def _draw_panel(self) -> None:
        assert self.screen and self.font and self.small

        panel = pygame.Rect(self._panel_x(), MARGIN, PANEL_W - 40, BOARD_SIZE)
        pygame.draw.rect(self.screen, (35, 35, 40), panel, border_radius=8)

        y = panel.y + 16
        title = self.font.render("Monopoly (Phase 2)", True, (240, 240, 240))
        self.screen.blit(title, (panel.x + 16, y))
        y += 36

        # status
        for p in self.sim.players:
            line = f"{p.name}: ${p.balance} | pos={p.position} | props={len(p.properties)}"
            txt = self.small.render(line, True, (220, 220, 220))
            self.screen.blit(txt, (panel.x + 16, y))
            y += 22

        y += 14
        self.screen.blit(self.small.render("Log:", True, (220, 220, 220)), (panel.x + 16, y))
        y += 22

        for line in self.log_lines[-18:]:
            self.screen.blit(self.small.render(line, True, (200, 200, 200)), (panel.x + 16, y))
            y += 20

        # buttons
        for b in self.buttons:
            col = (90, 160, 90) if b.enabled else (80, 80, 80)
            pygame.draw.rect(self.screen, col, b.rect, border_radius=6)
            pygame.draw.rect(self.screen, (0, 0, 0), b.rect, 2, border_radius=6)
            label = self.small.render(b.text, True, (0, 0, 0))
            self.screen.blit(label, (b.rect.x + 10, b.rect.y + 10))
