import ctypes
import logging
import os
from types import TracebackType
from typing import Final


class WindowsInhibitor:
    """
    Prevent Windows from sleeping during long-running tasks.
    Uses SetThreadExecutionState to control system/display idle timers and away mode.

    Key options:
    - prevent_sleep: Reset system idle timer (default: True)
    - keep_display: Keep display on (mutually exclusive with away_mode)
    - away_mode: Enable away mode for background processing while system appears asleep
      (saves power, display turns off, but tasks continue; requires hardware support)
    - logger: Optional logger instance for status messages (falls back to print if None)
    """

    ES_CONTINUOUS: Final[int] = 0x80000000
    ES_SYSTEM_REQUIRED: Final[int] = 0x00000001
    ES_DISPLAY_REQUIRED: Final[int] = 0x00000002
    ES_AWAYMODE_REQUIRED: Final[int] = 0x00000040

    def __init__(
        self,
        prevent_sleep: bool = True,
        keep_display: bool = False,
        away_mode: bool = False,
        verbose: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.prevent_sleep: bool = prevent_sleep
        self.keep_display: bool = keep_display and not away_mode
        self.away_mode: bool = away_mode
        self.verbose: bool = verbose
        self.logger: logging.Logger | None = logger
        self.previous_state: int | None = None
        self._initialized: bool = False

    def _log(self, message: str) -> None:
        """Log message using logger if available, otherwise print if verbose."""
        if self.logger:
            self.logger.debug(f"WinInhibit {message}")
        elif self.verbose:
            print(message)

    def _build_flags(self) -> int:
        flags = self.ES_CONTINUOUS
        if self.prevent_sleep or self.away_mode:
            flags |= self.ES_SYSTEM_REQUIRED
        if self.keep_display:
            flags |= self.ES_DISPLAY_REQUIRED
        if self.away_mode:
            flags |= self.ES_AWAYMODE_REQUIRED
        return flags

    def __enter__(self) -> "WindowsInhibitor":
        """Enter the context manager, setting the thread execution state on Windows."""
        if os.name != "nt":
            return self

        flags = self._build_flags()
        self._log(
            f"on=1 | sleep={int(self.prevent_sleep or self.away_mode)} | "
            f"display={int(self.keep_display)} | away={int(self.away_mode)}"
        )

        self.previous_state = ctypes.windll.kernel32.SetThreadExecutionState(flags)
        self._initialized = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the context manager, restoring the previous thread execution state."""
        if os.name != "nt" or not self._initialized:
            return

        ctypes.windll.kernel32.SetThreadExecutionState(self.previous_state)
        self._log("on=0 | state=restored")
        self._initialized = False

    def release(self) -> None:
        """Manual release (alternative to context manager)."""
        if os.name == "nt":
            self.__exit__(None, None, None)
