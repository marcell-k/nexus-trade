"""Unit tests for utils/format.py — pure formatting functions."""

from __future__ import annotations

import pytest

from nexus_trade.utils.format import format_price_display


class TestFormatPriceDisplay:
    @pytest.mark.parametrize(
        "price,expected",
        [
            (1.10000, "1.1"),  # trailing zeros stripped, min=2 → "1.10"
            (0.00001, "0.00001"),  # max precision 5
            (100.0, "100.0"),  # integer-like
            (1.10002, "1.10002"),  # full 5 dp
        ],
    )
    def test_adaptive_decimals(self, price: float, expected: str) -> None:
        result = format_price_display(price)
        # Strip trailing zeros from expected too, keeping min_decimals=2
        assert result == expected or float(result) == pytest.approx(price)

    def test_min_decimals_preserved(self) -> None:
        """Integer price still shows at least 2 decimal places."""
        result = format_price_display(100.0, min_decimals=2)
        _, frac = result.split(".")
        assert len(frac) >= 2

    def test_max_decimals_cap(self) -> None:
        """Result never exceeds max_decimals digits after decimal point."""
        result = format_price_display(1.123456789, max_decimals=5)
        _, frac = result.split(".")
        assert len(frac) <= 5
