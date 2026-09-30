"""Drawing helpers for documentation figures (requires matplotlib, a dev dependency)."""

import matplotlib.pyplot as plt

BOARD_COLOR = '#e3c16f'
LINE_COLOR = '#3b2f1e'


def draw_board(ax: plt.Axes, size: int, moves: list[int], numbers: bool = True, highlight_last: bool = True):
    """Draw a board with the stones of `moves` (black plays first), optionally numbered in play order."""
    ax.set_facecolor(BOARD_COLOR)
    for i in range(size):
        ax.plot([0, size - 1], [i, i], color=LINE_COLOR, lw=0.8, zorder=1)
        ax.plot([i, i], [0, size - 1], color=LINE_COLOR, lw=0.8, zorder=1)

    for n, pos in enumerate(moves):
        y, x = divmod(pos, size)
        black = n % 2 == 0
        ax.add_patch(plt.Circle((x, y), 0.45, facecolor='#111111' if black else '#fafafa',
                                edgecolor='#111111', lw=0.8, zorder=2))
        if numbers:
            ax.text(x, y, str(n + 1), ha='center', va='center', fontsize=7 if n < 99 else 5.5,
                    color='#fafafa' if black else '#111111', zorder=3)
    if highlight_last and moves:
        y, x = divmod(moves[-1], size)
        ax.add_patch(plt.Circle((x, y), 0.5, facecolor='none', edgecolor='#e34948', lw=2, zorder=4))

    labels = 'ABCDEFGHJKLMNOPQRST'[:size]  # Go-style column letters (no I)
    ax.set_xticks(range(size), labels)
    ax.set_yticks(range(size), [str(size - i) for i in range(size)])
    ax.tick_params(length=0, labelsize=8, colors='#52514e')
    ax.set_xlim(-0.7, size - 0.3)
    ax.set_ylim(size - 0.3, -0.7)  # row 0 at the top, like print(env)
    ax.set_aspect('equal')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
