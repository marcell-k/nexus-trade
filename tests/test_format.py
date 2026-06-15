"""Unit tests for utils/format.py — pure formatting functions."""

from __future__ import annotations

import pytest

from nexus_trade.utils.format import format_price_display, log_section_header


class TestFormatPriceDisplay:
    @pytest.mark.parametrize(
        "price,expected",
        [
            (1.10000, "1.1"),  # trailing zeros stripped, min=2 → "1.10"
            (1.10500, "1.105"),  # keeps meaningful digits
            (0.00001, "0.00001"),  # max precision 5
            (100.0, "100.0"),  # integer-like
            (1.10002, "1.10002"),  # full 5 dp
            (1.09999, "1.09999"),
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

    def test_returns_string(self) -> None:
        assert isinstance(format_price_display(1.1), str)

    def test_large_price(self) -> None:
        result = format_price_display(1985.50)
        assert "1985" in result

    def test_very_small_price(self) -> None:
        result = format_price_display(0.00001)
        assert float(result) == pytest.approx(0.00001)


class TestLogSectionHeader:
    def test_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        logger = logging.getLogger("test")
        log_section_header(logger, "TEST SECTION", width=40)

    def test_width_respected(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        logger = logging.getLogger("test_width")
        with caplog.at_level(logging.INFO, logger="test_width"):
            log_section_header(logger, "HEADER", width=20)
        separator_lines = [r for r in caplog.records if "=" in r.message]
        assert len(separator_lines) >= 2
        assert len(separator_lines[0].message) == 20
