"""Live display for Q-learning execution phase."""

import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.align import Align
from rich.box import DOUBLE, HEAVY, ROUNDED
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .themes import DEFAULT_THEME, TUITheme


class LiveDisplay:
    """Real-time live display for Q-learning request generation phase."""

    # Maximum individual status codes to show before grouping
    MAX_INDIVIDUAL_CODES = 8

    def __init__(
        self,
        time_duration: int,
        total_operations: int,
        theme: TUITheme = DEFAULT_THEME,
        width: int = 100,
    ):
        self.console = Console(force_terminal=True, width=width)
        self.theme = theme
        self.width = width
        self.time_duration = time_duration
        self.total_operations = total_operations

        # State tracking
        self.start_time: float = 0.0
        self.current_operation: str = ""
        self.responses: Counter = Counter()
        self.unique_errors: int = 0
        self.successful_operations: Set[str] = set()
        self.input_tokens: int = 0
        self.output_tokens: int = 0

        # Live display instance
        self._live: Optional[Live] = None

    def _format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _format_time_verbose(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        return " ".join(parts)

    def _create_time_panel(self) -> Panel:
        """Create the time elapsed/remaining panel with enhanced visuals."""
        elapsed = time.time() - self.start_time
        remaining = max(0, self.time_duration - elapsed)
        progress_pct = min(100, (elapsed / self.time_duration) * 100)
        remaining_pct = 100 - progress_pct

        # Calculate progress bar dimensions
        bar_width = min(self.width - 30, 70)
        filled = int((progress_pct / 100) * bar_width)
        empty = bar_width - filled

        # Create animated progress bar with gradient effect (based on % complete)
        if progress_pct < 25:
            fill_char = "━"
            fill_color = self.theme.info
        elif progress_pct < 50:
            fill_char = "━"
            fill_color = self.theme.primary
        elif progress_pct < 75:
            fill_char = "━"
            fill_color = self.theme.warning
        else:
            fill_char = "━"
            fill_color = self.theme.success

        # Add pulsing effect at the end of the progress bar
        if filled > 0 and empty > 0:
            bar = (
                f"[{fill_color}]{fill_char * (filled - 1)}[/{fill_color}]"
                f"[bold {fill_color}]▶[/bold {fill_color}]"
                f"[{self.theme.progress_remaining}]{'─' * empty}[/{self.theme.progress_remaining}]"
            )
        elif filled > 0:
            bar = f"[{fill_color}]{fill_char * filled}[/{fill_color}]"
        else:
            bar = f"[{self.theme.progress_remaining}]{'─' * bar_width}[/{self.theme.progress_remaining}]"

        # Time display with clear labels
        elapsed_str = self._format_time(elapsed)
        remaining_str = self._format_time(remaining)

        # Remaining time color based on percentage remaining (not absolute time)
        # Red: < 5% remaining, Yellow: < 20% remaining, Normal: >= 20%
        remaining_color = self.theme.error if remaining_pct < 5 else (
            self.theme.warning if remaining_pct < 20 else self.theme.text
        )

        # Build time display
        time_line = Text()
        time_line.append("  Time Elapsed: ", style=self.theme.text_dim)
        time_line.append(elapsed_str, style=f"bold {self.theme.primary}")
        time_line.append("  │  ", style=self.theme.text_dim)
        time_line.append("Time Remaining: ", style=self.theme.text_dim)
        time_line.append(remaining_str, style=f"bold {remaining_color}")

        # Progress percentage display
        pct_color = fill_color
        progress_line = Text()
        progress_line.append(f"  {bar} ", style="")
        progress_line.append(f"{progress_pct:5.1f}%", style=f"bold {pct_color}")

        return Panel(
            Group(
                Align.center(time_line),
                Text(),
                Align.center(progress_line),
            ),
            box=ROUNDED,
            border_style=self.theme.accent,
            title="[bold]Progress[/bold]",
            title_align="left",
            padding=(0, 1),
        )

    def _group_status_codes(self) -> List[Tuple[str, int, str]]:
        """Group status codes by category when there are many.

        Returns list of (label, count, color) tuples.
        """
        if len(self.responses) <= self.MAX_INDIVIDUAL_CODES:
            # Show individual codes
            result = []
            for code in sorted(self.responses.keys()):
                count = self.responses[code]
                color = self.theme.get_status_color(code)
                result.append((str(code), count, color))
            return result

        # Group by category
        categories = {
            "2xx Success": (0, self.theme.status_2xx),
            "3xx Redirect": (0, self.theme.status_3xx),
            "4xx Client Error": (0, self.theme.status_4xx),
            "5xx Server Error": (0, self.theme.status_5xx),
            "Other": (0, self.theme.text_dim),
        }

        for code, count in self.responses.items():
            if 200 <= code < 300:
                categories["2xx Success"] = (categories["2xx Success"][0] + count, categories["2xx Success"][1])
            elif 300 <= code < 400:
                categories["3xx Redirect"] = (categories["3xx Redirect"][0] + count, categories["3xx Redirect"][1])
            elif 400 <= code < 500:
                categories["4xx Client Error"] = (categories["4xx Client Error"][0] + count, categories["4xx Client Error"][1])
            elif 500 <= code < 600:
                categories["5xx Server Error"] = (categories["5xx Server Error"][0] + count, categories["5xx Server Error"][1])
            else:
                categories["Other"] = (categories["Other"][0] + count, categories["Other"][1])

        # Return non-zero categories
        result = []
        for label, (count, color) in categories.items():
            if count > 0:
                result.append((label, count, color))
        return result

    def _create_status_panel(self) -> Panel:
        """Create the status code distribution panel with smart grouping."""
        status_data = self._group_status_codes()

        if not status_data:
            return Panel(
                Align.center(Text("No requests sent yet...", style=self.theme.text_dim)),
                title="[bold]Status Code Distribution[/bold]",
                title_align="left",
                box=ROUNDED,
                border_style=self.theme.secondary,
                padding=(0, 1),
            )

        status_table = Table(
            box=None,
            show_header=True,
            header_style=f"bold {self.theme.text}",
            padding=(0, 1),
            expand=True,
        )
        status_table.add_column("Status", width=18, justify="left")
        status_table.add_column("Distribution", min_width=30)
        status_table.add_column("Count", width=12, justify="right")

        max_count = max(count for _, count, _ in status_data) if status_data else 1
        total_requests = sum(count for _, count, _ in status_data)

        for label, count, color in status_data:
            # Calculate bar width proportionally
            bar_width = min(40, max(1, int((count / max_count) * 40)))
            pct = (count / total_requests) * 100 if total_requests > 0 else 0

            # Create visual bar
            bar = f"[{color}]{'█' * bar_width}{'░' * (40 - bar_width)}[/{color}]"

            status_table.add_row(
                f"[{color}]{label}[/{color}]",
                bar,
                f"[{color}]{count:,}[/{color}] [dim]({pct:.1f}%)[/dim]",
            )

        return Panel(
            status_table,
            title="[bold]Status Code Distribution[/bold]",
            title_align="left",
            box=ROUNDED,
            border_style=self.theme.secondary,
            padding=(0, 1),
        )

    def _create_stats_panel(self) -> Panel:
        """Create the statistics panel."""
        stats_table = Table(box=None, show_header=False, padding=(0, 2), expand=True)
        stats_table.add_column("Metric", style=self.theme.text_dim, ratio=2)
        stats_table.add_column("Value", style=f"bold {self.theme.text}", justify="right", ratio=1)

        # Calculate statistics
        total_requests = sum(self.responses.values())
        success_pct = (len(self.successful_operations) / max(self.total_operations, 1)) * 100
        success_color = (
            self.theme.success if success_pct >= 70 else
            self.theme.warning if success_pct >= 40 else
            self.theme.error
        )

        # Requests per second
        elapsed = time.time() - self.start_time
        rps = total_requests / max(elapsed, 1)

        stats_table.add_row(
            "Successfully Processed (2xx) Operations:",
            f"[{success_color}]{len(self.successful_operations):,} / {self.total_operations}[/{success_color}]",
        )
        stats_table.add_row(
            "Operation Coverage:",
            f"[{success_color}]{success_pct:.1f}%[/{success_color}]",
        )
        stats_table.add_row(
            "Unique Server Errors (5xx):",
            f"[{self.theme.error if self.unique_errors > 0 else self.theme.success}]{self.unique_errors}[/]",
        )
        stats_table.add_row(
            "Total Requests Sent:",
            f"[{self.theme.info}]{total_requests:,}[/{self.theme.info}]",
        )
        stats_table.add_row(
            "Requests/Second:",
            f"[{self.theme.info}]{rps:.1f}[/{self.theme.info}]",
        )

        return Panel(
            stats_table,
            title="[bold]Statistics[/bold]",
            title_align="left",
            box=ROUNDED,
            border_style=self.theme.info,
            padding=(0, 1),
        )

    def _create_operation_panel(self) -> Panel:
        """Create the current operation panel."""
        op_text = Text()
        op_text.append("▶ ", style=f"bold {self.theme.success}")
        op_text.append(self.current_operation or "Initializing...", style=f"bold {self.theme.primary}")

        return Panel(
            Align.center(op_text),
            title="[bold]Current Operation[/bold]",
            title_align="left",
            box=ROUNDED,
            border_style=self.theme.accent,
            padding=(0, 1),
        )

    def _create_cost_panel(self) -> Panel:
        """Create the LLM cost tracking panel."""
        # Rough cost estimate (adjust based on your model's pricing)
        # Using approximate rates: input $0.0001/1K tokens, output $0.0002/1K tokens
        estimated_cost = (self.input_tokens * 0.0001 + self.output_tokens * 0.0002) / 1000

        cost_text = Text()
        cost_text.append("💰 ", style="")
        cost_text.append("LLM Cost: ", style=self.theme.text_dim)

        cost_color = (
            self.theme.success if estimated_cost < 0.50 else
            self.theme.warning if estimated_cost < 2.0 else
            self.theme.error
        )
        cost_text.append(f"${estimated_cost:.2f}", style=f"bold {cost_color}")
        cost_text.append(" USD", style=self.theme.text_dim)

        # Token count
        cost_text.append("  │  ", style=self.theme.text_dim)
        cost_text.append(f"Tokens: {self.input_tokens + self.output_tokens:,}", style=self.theme.text_dim)

        return Panel(
            Align.center(cost_text),
            box=ROUNDED,
            border_style=self.theme.warning,
            padding=(0, 0),
        )

    def _generate_display(self) -> Panel:
        """Generate the complete live display."""
        # Header
        header = Text()
        header.append("⚡ ", style=f"bold {self.theme.warning}")
        header.append("REQUEST GENERATION", style=f"bold {self.theme.primary}")
        header.append(" ⚡", style=f"bold {self.theme.warning}")

        # Combine all panels
        content = Group(
            self._create_time_panel(),
            Text(),
            self._create_stats_panel(),
            Text(),
            self._create_status_panel(),
            Text(),
            self._create_operation_panel(),
            Text(),
            self._create_cost_panel(),
        )

        return Panel(
            content,
            title=header,
            title_align="center",
            box=DOUBLE,
            border_style=self.theme.primary,
            padding=(1, 2),
        )

    def start(self):
        """Start the live display."""
        self.start_time = time.time()
        self._live = Live(
            self._generate_display(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self):
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def update(
        self,
        current_operation: str,
        responses: Counter,
        unique_errors: int,
        successful_operations: Set[str],
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """Update the live display with new data."""
        self.current_operation = current_operation
        self.responses = responses
        self.unique_errors = unique_errors
        self.successful_operations = successful_operations
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

        if self._live:
            self._live.update(self._generate_display())

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class ProgressDisplay:
    """Progress display for graph construction and Q-table initialization."""

    def __init__(self, theme: TUITheme = DEFAULT_THEME, width: int = 100):
        self.console = Console(force_terminal=True, width=width)
        self.theme = theme
        self.width = width

    def create_progress_bar(
        self,
        description: str,
        total: int,
        transient: bool = False,
    ) -> Progress:
        """Create a styled progress bar."""
        progress = Progress(
            SpinnerColumn(style=self.theme.primary),
            TextColumn(f"[{self.theme.text}]{description}[/{self.theme.text}]"),
            BarColumn(
                bar_width=40,
                complete_style=self.theme.progress_complete,
                finished_style=self.theme.success,
            ),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn(f"[{self.theme.text_dim}]•[/{self.theme.text_dim}]"),
            TextColumn(f"[{self.theme.info}]{{task.completed}}/{{task.total}}[/{self.theme.info}]"),
            console=self.console,
            transient=transient,
        )
        return progress

    def track_operation(self, description: str, total: int):
        """Create a tracking context for operations."""
        return self.create_progress_bar(description, total).__enter__()
