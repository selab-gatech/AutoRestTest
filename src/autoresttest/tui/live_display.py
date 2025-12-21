"""Live display for Q-learning execution phase."""

import time
from collections import Counter
from typing import Any, Dict, Optional, Set

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

    def _create_time_panel(self) -> Panel:
        """Create the time elapsed/remaining panel."""
        elapsed = time.time() - self.start_time
        remaining = max(0, self.time_duration - elapsed)
        progress_pct = min(100, (elapsed / self.time_duration) * 100)

        # Time display table
        time_table = Table(box=None, show_header=False, padding=(0, 2))
        time_table.add_column("Label", style=self.theme.text_dim)
        time_table.add_column("Value", style=f"bold {self.theme.text}")
        time_table.add_column("Label2", style=self.theme.text_dim)
        time_table.add_column("Value2", style=f"bold {self.theme.text}")

        elapsed_str = self._format_time(elapsed)
        remaining_str = self._format_time(remaining)

        time_table.add_row(
            "Time Elapsed:",
            f"[{self.theme.primary}]{elapsed_str}[/{self.theme.primary}]",
            "Time Remaining:",
            f"[{self.theme.warning if remaining < 60 else self.theme.text}]{remaining_str}[/]",
        )

        # Progress bar
        bar_width = self.width - 20
        filled = int((progress_pct / 100) * bar_width)
        bar = (
            f"[{self.theme.progress_complete}]{'━' * filled}[/{self.theme.progress_complete}]"
            f"[{self.theme.progress_remaining}]{'─' * (bar_width - filled)}[/{self.theme.progress_remaining}]"
        )
        progress_text = Text.assemble(
            ("  ", ""),
            (bar, ""),
            (f" {progress_pct:.1f}%", f"bold {self.theme.primary}"),
        )

        return Panel(
            Group(Align.center(time_table), Align.center(progress_text)),
            box=ROUNDED,
            border_style=self.theme.accent,
            padding=(0, 1),
        )

    def _create_status_panel(self) -> Panel:
        """Create the status code distribution panel."""
        status_table = Table(
            box=None,
            show_header=True,
            header_style=f"bold {self.theme.text}",
            padding=(0, 1),
        )
        status_table.add_column("Code", width=5, justify="center")
        status_table.add_column("Distribution", width=50)
        status_table.add_column("Count", width=10, justify="right")

        max_count = max(self.responses.values()) if self.responses else 1

        # Group status codes by category
        categories = {
            "2xx": ([code for code in self.responses if 200 <= code < 300], self.theme.status_2xx),
            "3xx": ([code for code in self.responses if 300 <= code < 400], self.theme.status_3xx),
            "4xx": ([code for code in self.responses if 400 <= code < 500], self.theme.status_4xx),
            "5xx": ([code for code in self.responses if 500 <= code < 600], self.theme.status_5xx),
        }

        for code in sorted(self.responses.keys()):
            count = self.responses[code]
            color = self.theme.get_status_color(code)
            bar_width = int((count / max_count) * 45)
            bar = f"[{color}]{'█' * bar_width}{'░' * (45 - bar_width)}[/{color}]"
            status_table.add_row(
                f"[{color}]{code}[/{color}]",
                bar,
                f"[{color}]{count:,}[/{color}]",
            )

        title = Text()
        title.append("Status Code Distribution", style=f"bold {self.theme.secondary}")

        return Panel(
            status_table if self.responses else Align.center(Text("No requests sent yet", style=self.theme.text_dim)),
            title=title,
            title_align="left",
            box=ROUNDED,
            border_style=self.theme.secondary,
            padding=(0, 1),
        )

    def _create_stats_panel(self) -> Panel:
        """Create the statistics panel."""
        stats_table = Table(box=None, show_header=False, padding=(0, 2))
        stats_table.add_column("Metric", style=self.theme.text_dim)
        stats_table.add_column("Value", style=f"bold {self.theme.text}", justify="right")

        # Calculate statistics
        total_2xx = sum(count for code, count in self.responses.items() if 200 <= code < 300)
        success_pct = (len(self.successful_operations) / max(self.total_operations, 1)) * 100
        success_color = (
            self.theme.success if success_pct >= 70 else
            self.theme.warning if success_pct >= 40 else
            self.theme.error
        )

        stats_table.add_row(
            "Successfully Processed (2xx) Operations:",
            f"[{success_color}]{len(self.successful_operations):,}[/{success_color}]",
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
            f"[{self.theme.info}]{sum(self.responses.values()):,}[/{self.theme.info}]",
        )

        return Panel(
            Align.center(stats_table),
            box=ROUNDED,
            border_style=self.theme.info,
            padding=(0, 1),
        )

    def _create_operation_panel(self) -> Panel:
        """Create the current operation panel."""
        op_text = Text()
        op_text.append("Current Operation: ", style=self.theme.text_dim)
        op_text.append(self.current_operation or "Initializing...", style=f"bold {self.theme.primary}")

        return Panel(
            Align.center(op_text),
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
        cost_text.append("[COST] ", style=f"bold {self.theme.warning}")
        cost_text.append("Total LLM usage: ", style=self.theme.text_dim)

        cost_color = self.theme.success if estimated_cost < 0.50 else self.theme.warning if estimated_cost < 2.0 else self.theme.error
        cost_text.append(f"${estimated_cost:.2f} USD", style=f"bold {cost_color}")

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
        header.append("[STATUS] ", style=f"bold {self.theme.warning}")
        header.append("Request Generation", style=f"bold {self.theme.primary}")

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
            title_align="left",
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
