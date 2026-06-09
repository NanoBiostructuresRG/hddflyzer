# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for hddflyzer.viz.colors."""

from hddflyzer.viz.colors import (
    TAG_COLORS,
    anabook_palette_8,
    anabook_palette_16,
    get_tag_color,
)


class TestColors:
    def test_palette_8_length(self):
        assert len(anabook_palette_8) == 8

    def test_palette_16_length(self):
        assert len(anabook_palette_16) == 16

    def test_palette_8_are_hex(self):
        for c in anabook_palette_8:
            assert c.startswith("#"), f"{c} is not a hex color"
            assert len(c) == 7

    def test_get_tag_color_known(self):
        assert get_tag_color("aocd") == TAG_COLORS["aocd"]

    def test_get_tag_color_unknown(self):
        assert get_tag_color("unknown_tag") == TAG_COLORS["default"]

    def test_get_tag_color_case_insensitive(self):
        assert get_tag_color("AOCD") == get_tag_color("aocd")
