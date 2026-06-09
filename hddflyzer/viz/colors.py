# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Color palettes for HDDFlyzer visualizations.

Palettes
--------
anabook_palette_8  : 8-color personal palette (soft, harmonious)
anabook_palette_16 : 16-color extended palette
husl_palette       : 8 vibrant equidistant scientific colors (HUSL)
"""

import seaborn as sns
import matplotlib.colors as mcolors

# AnaBook Palette — 8 colors
anabook_palette_8 = [
    "#8fd7d7",  # [0] Light Blue
    "#00b0be",  # [1] Med Blue
    "#ff8ca1",  # [2] Light Pink
    "#f45f74",  # [3] Med Pink
    "#bdd373",  # [4] Light Green
    "#98c127",  # [5] Med Green
    "#ffcd8e",  # [6] Light Orange
    "#ffc255",  # [7] Med Orange
]

# AnaBook Palette — 16 colors
anabook_palette_16 = [
    "#8fd7d7",  # [0]  Light Blue
    "#00b0be",  # [1]  Med Blue
    "#ff8ca1",  # [2]  Light Pink
    "#f45f74",  # [3]  Med Pink
    "#bdd373",  # [4]  Light Green
    "#98c127",  # [5]  Med Green
    "#ffcd8e",  # [6]  Light Orange
    "#ffc255",  # [7]  Med Orange
    "#c9a7d0",  # [8]  Light Purple
    "#a885c1",  # [9]  Med Purple
    "#ffb380",  # [10] Light Coral
    "#ff8c69",  # [11] Med Coral
    "#85d4c9",  # [12] Light Teal
    "#36a89d",  # [13] Med Teal
    "#e6d88c",  # [14] Light Yellow
    "#d4c15a",  # [15] Med Yellow
]

# HUSL Palette — 8 vibrant equidistant colors
husl_palette = [mcolors.rgb2hex(color) for color in sns.husl_palette(8)]

# Tag-specific colors (used across plot scripts)
TAG_COLORS = {
    "aocd":     "#00b0be",  # Med Blue
    "dianatdb": "#f45f74",  # Med Pink
    "default":  "#1f77b4",  # Matplotlib blue fallback
}


def get_tag_color(tag: str) -> str:
    """Return the canonical color for a given dataset tag."""
    return TAG_COLORS.get(tag.lower(), TAG_COLORS["default"])
