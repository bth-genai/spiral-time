"""Render the exact historical-return mechanism figure."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


CELLS = (
    ("bodies_4_capacity_1.44", "4 bodies\ncapacity 1.44"),
    ("bodies_6_capacity_1.44", "6 bodies\ncapacity 1.44"),
    ("bodies_6_capacity_2.40", "6 bodies\ncapacity 2.40"),
)
POLICIES = ("blocked", "reverse", "interleaved", "shuffled")
COLORS = {
    "blocked": "#167D6B",
    "reverse": "#C4493D",
    "interleaved": "#33658A",
    "shuffled": "#D39B2A",
}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    paths = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    )
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _centered_text(draw, x: float, y: float, value: str, *, fill: str, font) -> None:
    box = draw.multiline_textbbox((0, 0), value, font=font, align="center", spacing=3)
    draw.multiline_text(
        (x - (box[2] - box[0]) / 2, y),
        value,
        fill=fill,
        font=font,
        align="center",
        spacing=3,
    )


def _bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(5000, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def _panel_axes(
    draw,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    title: str,
    subtitle: str,
    y_max: float,
    tick_step: float,
    threshold: bool = True,
) -> None:
    draw.text((left, 122), title, fill="#20252B", font=_font(23, True))
    draw.text((left, 153), subtitle, fill="#566573", font=_font(16))
    ticks = np.arange(0.0, y_max + 1.0e-12, tick_step)
    for tick in ticks:
        y = top + height - float(tick) / y_max * height
        draw.line((left, y, left + width, y), fill="#DDE2E6", width=1)
        draw.text((left - 48, y - 9), f"{tick:.2f}", fill="#566573", font=_font(14))
    draw.line((left, top, left, top + height), fill="#20252B", width=2)
    draw.line((left, top + height, left + width, top + height), fill="#20252B", width=2)
    if threshold:
        threshold_y = top + height - 0.025 / y_max * height
        for x in range(left, left + width, 13):
            draw.line(
                (x, threshold_y, min(x + 6, left + width), threshold_y),
                fill="#7F8C8D",
                width=2,
            )


def _error_bar(draw, x, estimate, low, high, *, top, height, y_max, color):
    y = lambda value: top + height - max(0.0, min(y_max, value)) / y_max * height
    y_estimate, y_low, y_high = y(estimate), y(low), y(high)
    draw.line((x, y_high, x, y_low), fill=color, width=3)
    draw.line((x - 7, y_high, x + 7, y_high), fill=color, width=3)
    draw.line((x - 7, y_low, x + 7, y_low), fill=color, width=3)
    draw.ellipse(
        (x - 6, y_estimate - 6, x + 6, y_estimate + 6),
        fill=color,
        outline="white",
        width=1,
    )


def draw_figure(comparison_path: Path, trial_path: Path, output_path: Path) -> None:
    comparisons = pd.read_csv(comparison_path)
    trials = pd.read_csv(trial_path)
    image = Image.new("RGB", (1800, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (68, 26),
        "The same present encounter returns differently through a changed history",
        fill="#17202A",
        font=_font(31, True),
    )
    draw.text(
        (68, 70),
        "Paired worlds reuse A and B, forcing, initial phases, noise, and probe horizon",
        fill="#566573",
        font=_font(18),
    )

    top, height, width = 195, 440, 455
    panel_lefts = (95, 675, 1255)
    panel_specs = (
        (
            "A  Intervening participation",
            "Absolute distributed-history minus passive-time difference",
            0.50,
            0.10,
        ),
        (
            "B  Historical order",
            "Within-world range across the same encounter tokens",
            0.14,
            0.02,
        ),
        (
            "C  Relational orientation",
            "Absolute tissue-only rotation minus unchanged relation",
            0.50,
            0.10,
        ),
    )
    for left, (title, subtitle, y_max, tick_step) in zip(panel_lefts, panel_specs):
        _panel_axes(
            draw,
            left=left,
            top=top,
            width=width,
            height=height,
            title=title,
            subtitle=subtitle,
            y_max=y_max,
            tick_step=tick_step,
        )

    offsets = (-24, -8, 8, 24)
    for cell_index, (cell, label) in enumerate(CELLS):
        center = panel_lefts[0] + 78 + cell_index * 150
        for policy, offset in zip(POLICIES, offsets):
            row = comparisons[
                comparisons["cell"].eq(cell)
                & comparisons["policy"].eq(policy)
                & comparisons["left_condition"].eq("distributed")
                & comparisons["right_condition"].eq("passive_elapsed")
            ].iloc[0]
            _error_bar(
                draw,
                center + offset,
                float(row["mean_absolute_carrying_difference"]),
                float(row["absolute_difference_low"]),
                float(row["absolute_difference_high"]),
                top=top,
                height=height,
                y_max=0.50,
                color=COLORS[policy],
            )
        _centered_text(
            draw,
            center,
            top + height + 18,
            label,
            fill="#424949",
            font=_font(14),
        )

    for cell_index, (cell, label) in enumerate(CELLS):
        center = panel_lefts[1] + 78 + cell_index * 150
        selected = trials[
            trials["cell"].eq(cell)
            & trials["condition"].isin(("distributed", "commutative"))
        ]
        for condition, offset, color in (
            ("distributed", -11, "#C4493D"),
            ("commutative", 11, "#566573"),
        ):
            pivot = selected[selected["condition"].eq(condition)].pivot(
                index="seed", columns="policy", values="carried"
            )
            ranges = (pivot.max(axis=1) - pivot.min(axis=1)).to_numpy(float)
            estimate, low, high = _bootstrap(ranges, 6610 + cell_index)
            _error_bar(
                draw,
                center + offset,
                estimate,
                low,
                high,
                top=top,
                height=height,
                y_max=0.14,
                color=color,
            )
        _centered_text(
            draw,
            center,
            top + height + 18,
            label,
            fill="#424949",
            font=_font(14),
        )

    for cell_index, (cell, label) in enumerate(CELLS):
        center = panel_lefts[2] + 78 + cell_index * 150
        for policy, offset in zip(POLICIES, offsets):
            row = comparisons[
                comparisons["cell"].eq(cell)
                & comparisons["policy"].eq(policy)
                & comparisons["left_condition"].eq("tissue_only_rotation")
                & comparisons["right_condition"].eq("distributed")
            ].iloc[0]
            _error_bar(
                draw,
                center + offset,
                float(row["mean_absolute_carrying_difference"]),
                float(row["absolute_difference_low"]),
                float(row["absolute_difference_high"]),
                top=top,
                height=height,
                y_max=0.50,
                color=COLORS[policy],
            )
        _centered_text(
            draw,
            center,
            top + height + 18,
            label,
            fill="#424949",
            font=_font(14),
        )

    legend_y = 748
    legend_items = [
        ("blocked", COLORS["blocked"]),
        ("reverse", COLORS["reverse"]),
        ("interleaved", COLORS["interleaved"]),
        ("shuffled", COLORS["shuffled"]),
        ("commutative null", "#566573"),
    ]
    x = 525
    for label, color in legend_items:
        draw.ellipse((x, legend_y, x + 12, legend_y + 12), fill=color)
        draw.text((x + 18, legend_y - 4), label, fill="#424949", font=_font(14))
        x += 88 + len(label) * 6
    draw.text(
        (1285, 708),
        "Co-transform control: maximum difference 3.7e-14",
        fill="#566573",
        font=_font(14),
    )
    draw.text(
        (108, 708),
        "Dashed line: predeclared material threshold 0.025",
        fill="#566573",
        font=_font(14),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    manifest_lines = []
    for path in sorted(output_path.parent.iterdir()):
        if path.is_file() and path.name != "MANIFEST.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_lines.append(f"{digest}  {path.name}")
    (output_path.parent / "MANIFEST.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparisons",
        type=Path,
        default=Path(
            "collective_historical_return_output/historical_return_comparisons.csv"
        ),
    )
    parser.add_argument(
        "--trials",
        type=Path,
        default=Path("collective_historical_return_output/historical_return_trials.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "collective_historical_return_output/fig_historical_return_controls.png"
        ),
    )
    args = parser.parse_args()
    draw_figure(args.comparisons, args.trials, args.output)


if __name__ == "__main__":
    main()
