"""
PolySwarm CLI theme — consistent styling across all commands.

Uses Rich library for terminal rendering with a cohesive visual identity:
amber/gold accents, clean layouts, visual hierarchy.
"""

from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
from rich.padding import Padding
from rich import box

# ═══════════════════════════════════════════
# Color palette
# ═══════════════════════════════════════════
COLORS = {
    "brand":      "#F59E0B",   # amber/gold — primary brand color
    "brand_dim":  "#B45309",   # darker amber
    "accent":     "#06B6D4",   # cyan
    "accent2":    "#8B5CF6",   # purple
    "positive":   "#22C55E",   # green
    "negative":   "#EF4444",   # red
    "warning":    "#EAB308",   # yellow
    "neutral":    "#6B7280",   # gray
    "text":       "#E5E7EB",   # light gray text
    "dim":        "#6B7280",   # dimmed text
    "muted":      "#374151",   # very dim
}

custom_theme = Theme({
    "brand":     f"bold {COLORS['brand']}",
    "accent":    COLORS["accent"],
    "accent2":   COLORS["accent2"],
    "positive":  COLORS["positive"],
    "negative":  COLORS["negative"],
    "warning":   COLORS["warning"],
    "info":      COLORS["accent"],
    "dimmed":    COLORS["dim"],
    "muted":     COLORS["muted"],
    "label":     f"bold {COLORS['text']}",
    "value":     f"bold {COLORS['brand']}",
})

console = Console(theme=custom_theme)

# ═══════════════════════════════════════════
# Box styles
# ═══════════════════════════════════════════
HEAVY_BOX = box.Box(
    "╔═╤╗\n"
    "║ │║\n"
    "╠═╪╣\n"
    "║ │║\n"
    "╠═╪╣\n"
    "╠═╪╣\n"
    "║ │║\n"
    "╚═╧╝\n"
)

# ═══════════════════════════════════════════
# Brand elements
# ═══════════════════════════════════════════

LOGO = """[bold #F59E0B]
  ██████╗  ██████╗ ██╗  ██╗   ██╗███████╗██╗    ██╗ █████╗ ██████╗ ███╗   ███╗
  ██╔══██╗██╔═══██╗██║  ╚██╗ ██╔╝██╔════╝██║    ██║██╔══██╗██╔══██╗████╗ ████║
  ██████╔╝██║   ██║██║   ╚████╔╝ ███████╗██║ █╗ ██║███████║██████╔╝██╔████╔██║
  ██╔═══╝ ██║   ██║██║    ╚██╔╝  ╚════██║██║███╗██║██╔══██║██╔══██╗██║╚██╔╝██║
  ██║     ╚██████╔╝███████╗██║   ███████║╚███╔███╔╝██║  ██║██║  ██║██║ ╚═╝ ██║
  ╚═╝      ╚═════╝ ╚══════╝╚═╝   ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝[/]"""

LOGO_SMALL = "[bold #F59E0B]>>> PolySwarm[/]"

DRAGONFLY = "[bold #F59E0B]<<<>>>[/]"


def header(title: str, subtitle: str = "", style: str = "brand"):
    """Print a branded header panel."""
    content = f"[bold #F59E0B]{DRAGONFLY}  {title}[/]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"
    console.print()
    console.print(Panel(
        content,
        border_style=COLORS["brand"],
        padding=(1, 2),
    ))


def section(title: str):
    """Print a section divider."""
    console.print()
    console.print(f"  [bold {COLORS['brand']}]{'━' * 3} {title} {'━' * (50 - len(title))}[/]")
    console.print()


def stat_card(label: str, value: str, color: str = "brand", detail: str = ""):
    """Single stat display."""
    line = f"  [{COLORS['dim']}]{label}[/]  [bold {COLORS.get(color, color)}]{value}[/]"
    if detail:
        line += f"  [{COLORS['dim']}]{detail}[/]"
    console.print(line)


def stat_row(stats: list[tuple[str, str, str]]):
    """Print a row of stats as: label value  |  label value  |  ..."""
    parts = []
    for label, value, color in stats:
        parts.append(f"[{COLORS['dim']}]{label}[/] [bold {COLORS.get(color, color)}]{value}[/]")
    console.print("  " + "  [dim]|[/dim]  ".join(parts))


def progress_bar(value: float, width: int = 20, filled_color: str = "brand", empty_char: str = "░") -> str:
    """Return a unicode progress bar string."""
    filled = int(value * width)
    bar_filled = "█" * filled
    bar_empty = empty_char * (width - filled)
    return f"[{COLORS.get(filled_color, filled_color)}]{bar_filled}[/][{COLORS['muted']}]{bar_empty}[/]"


def sentiment_bar(value: float, width: int = 16) -> str:
    """Return a centered sentiment bar: red ◄══|══► green."""
    center = width // 2
    filled = int(abs(value) * center)
    if value >= 0:
        left = f"[{COLORS['muted']}]{'░' * center}[/]"
        right_fill = "█" * filled
        right_empty = "░" * (center - filled)
        right = f"[{COLORS['positive']}]{right_fill}[/][{COLORS['muted']}]{right_empty}[/]"
    else:
        left_empty = "░" * (center - filled)
        left_fill = "█" * filled
        left = f"[{COLORS['muted']}]{left_empty}[/][{COLORS['negative']}]{left_fill}[/]"
        right = f"[{COLORS['muted']}]{'░' * center}[/]"
    return f"{left}[dim]|[/dim]{right}"


def probability_color(p: float) -> str:
    """Return a color based on probability value."""
    if p >= 0.8:
        return COLORS["positive"]
    elif p >= 0.6:
        return "#22D3EE"  # light cyan
    elif p >= 0.4:
        return COLORS["brand"]
    elif p >= 0.2:
        return COLORS["warning"]
    else:
        return COLORS["negative"]


def edge_color(edge: float) -> str:
    """Color for edge values."""
    if abs(edge) < 0.02:
        return COLORS["dim"]
    return COLORS["positive"] if edge > 0 else COLORS["negative"]


def brier_color(score: float) -> str:
    """Color for Brier scores."""
    if score < 0.1:
        return COLORS["positive"]
    elif score < 0.2:
        return COLORS["warning"]
    return COLORS["negative"]


def quality_label(score: float) -> str:
    """Quality label for Brier scores."""
    if score < 0.05:
        return f"[{COLORS['positive']}]Excellent[/]"
    elif score < 0.1:
        return f"[{COLORS['positive']}]Good[/]"
    elif score < 0.2:
        return f"[{COLORS['warning']}]Fair[/]"
    return f"[{COLORS['negative']}]Poor[/]"


def status_badge(available: bool, has_key: bool, requires_key: str | None = None) -> str:
    """Status badge for data sources."""
    if available:
        return f"[{COLORS['positive']}]● LIVE[/]"
    elif not has_key:
        return f"[{COLORS['warning']}]○ KEY[/]"
    return f"[{COLORS['negative']}]○ OFF[/]"


def category_style(category: str) -> str:
    """Color for data source categories."""
    cat_colors = {
        "market": "#22D3EE",
        "derivatives": "#8B5CF6",
        "onchain": "#F97316",
        "defi": "#10B981",
        "sentiment": "#EC4899",
        "social": "#06B6D4",
        "prediction_markets": "#F59E0B",
    }
    color = cat_colors.get(category, COLORS["dim"])
    return f"[{color}]{category}[/]"


def footer():
    """Print a branded footer."""
    console.print()
    console.print(f"  [{COLORS['dim']}]{'─' * 56}[/]")
    console.print(f"  [{COLORS['dim']}]{DRAGONFLY} PolySwarm v0.8.0  ·  github.com/defidaddydavid/polyswarm[/]")
    console.print()
