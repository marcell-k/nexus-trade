"""Unit tests for nexus_trade.core.constants — string_to_timeframe, TIMEFRAME_TO_MINUTES."""

from __future__ import annotations

import math

import pytest

from nexus_trade.core.constants import (
    TIMEFRAME_STRING_MAP,
    TIMEFRAME_TO_MINUTES,
    TimeFrame,
    string_to_timeframe,
)

#  string_to_timeframe 


class TestStringToTimeframe:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("M1", TimeFrame.M1),
            ("M5", TimeFrame.M5),
            ("M15", TimeFrame.M15),
            ("M30", TimeFrame.M30),
            ("H1", TimeFrame.H1),
            ("H4", TimeFrame.H4),
            ("D1", TimeFrame.D1),
            ("W1", TimeFrame.W1),
            ("MN1", TimeFrame.MN1),
        ],
    )
    def test_known_strings(self, s: str, expected: TimeFrame) -> None:
        assert string_to_timeframe(s) == expected

    @pytest.mark.parametrize("s", ["m1", "m15", "h1", "d1", "mn1"])
    def test_case_insensitive(self, s: str) -> None:
        assert string_to_timeframe(s) is not None

    def test_unknown_returns_none(self) -> None:
        assert string_to_timeframe("X9") is None

    def test_empty_string_returns_none(self) -> None:
        assert string_to_timeframe("") is None


#  TIMEFRAME_TO_MINUTES 


class TestTimeframeToMinutes:
    def test_m1_is_1(self) -> None:
        assert TIMEFRAME_TO_MINUTES[TimeFrame.M1] == 1

    def test_h1_is_60(self) -> None:
        assert TIMEFRAME_TO_MINUTES[TimeFrame.H1] == 60

    def test_h4_is_240(self) -> None:
        assert TIMEFRAME_TO_MINUTES[TimeFrame.H4] == 240

    def test_d1_is_1440(self) -> None:
        assert TIMEFRAME_TO_MINUTES[TimeFrame.D1] == 1440

    def test_all_timeframes_covered(self) -> None:
        """Every timeframe in the string map has a minutes entry."""
        for tf in TIMEFRAME_STRING_MAP.values():
            assert tf in TIMEFRAME_TO_MINUTES, f"{tf!r} missing from TIMEFRAME_TO_MINUTES"

    def test_all_values_positive(self) -> None:
        for tf, minutes in TIMEFRAME_TO_MINUTES.items():
            assert minutes > 0, f"{tf!r} has non-positive minutes: {minutes}"

    def test_ascending_order(self) -> None:
        """Spot-check ordering: M1 < M5 < M15 < M30 < H1 < H4 < D1 < W1 < MN1."""
        ordered = [
            TimeFrame.M1,
            TimeFrame.M5,
            TimeFrame.M15,
            TimeFrame.M30,
            TimeFrame.H1,
            TimeFrame.H4,
            TimeFrame.D1,
            TimeFrame.W1,
            TimeFrame.MN1,
        ]
        minutes = [TIMEFRAME_TO_MINUTES[tf] for tf in ordered]
        assert minutes == sorted(minutes), "TIMEFRAME_TO_MINUTES ordering broken"
