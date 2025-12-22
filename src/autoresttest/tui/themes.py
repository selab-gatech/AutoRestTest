"""Theme definitions for AutoRestTest TUI."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TUITheme:
    """Color and style definitions for the TUI."""

    # Primary colors
    primary: str = "#00D4FF"  # Cyan blue
    secondary: str = "#FF6B6B"  # Coral red
    accent: str = "#4ECDC4"  # Teal
    success: str = "#2ECC71"  # Green
    warning: str = "#F1C40F"  # Yellow
    error: str = "#E74C3C"  # Red
    info: str = "#3498DB"  # Blue

    # Text colors
    text: str = "#FFFFFF"
    text_dim: str = "#7F8C8D"
    text_muted: str = "#566573"

    # Background shades
    bg_dark: str = "#0D1117"
    bg_panel: str = "#161B22"
    bg_highlight: str = "#21262D"

    # Status code colors
    status_2xx: str = "#2ECC71"  # Green
    status_3xx: str = "#3498DB"  # Blue
    status_4xx: str = "#F39C12"  # Orange
    status_5xx: str = "#E74C3C"  # Red

    # Progress bar colors
    progress_complete: str = "#00D4FF"
    progress_remaining: str = "#2C3E50"

    # Symbols (plain characters - styling applied at usage site)
    symbol_success: str = "✓"
    symbol_error: str = "✗"
    symbol_warning: str = "⚠"
    symbol_info: str = "ℹ"
    symbol_arrow: str = "→"
    symbol_bullet: str = "•"
    symbol_progress: str = "⏳"

    # Symbol colors (for applying styles at usage site)
    symbol_success_color: str = "green"
    symbol_error_color: str = "red"
    symbol_warning_color: str = "yellow"
    symbol_info_color: str = "blue"
    symbol_arrow_color: str = "cyan"
    symbol_bullet_color: str = "dim"
    symbol_progress_color: str = "cyan"

    def get_status_color(self, status_code: int) -> str:
        """Return the appropriate color for an HTTP status code."""
        if 200 <= status_code < 300:
            return self.status_2xx
        elif 300 <= status_code < 400:
            return self.status_3xx
        elif 400 <= status_code < 500:
            return self.status_4xx
        elif 500 <= status_code < 600:
            return self.status_5xx
        return self.text_dim


# Default theme instance
DEFAULT_THEME = TUITheme()
