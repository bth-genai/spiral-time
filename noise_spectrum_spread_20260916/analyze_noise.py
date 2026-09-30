"""Cluster-aware analysis and figures for O5b."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHECKPOINTS = ("formation", "withdrawal", "return_6")
HISTORIES = ("alternating", "blocked")
EDGES = ("AB", "BC")

AMPLITUDE_CONDITIONS = (
    "white_0000",
    "white_0125",
    "white_0250",
    "white_0500",
    "white_1000",
)
STEP_SPECTRAL_CONDITIONS = (
    "blue_0250",
    "white_0250",
    "pink_0250",
    "ou_0250",
    "brown_0250",
)
DISPLACEMENT_SPECTRAL_CONDITIONS = (
    "blue_displacement_0250",
    "white_0250",
    "pink_displacement_0250",
    "ou_displacement_0250",
    "brown_displacement_0250",
)


def cell(recipient: dict, checkpoint: str, history: str, edge: str) -> dict:
    return recipient["cells"][checkpoint][history][edge]


def recipient_metric(recipient: dict, metric: str) -> float:
    values = []
    for checkpoint in CHECKPOINTS:
        for history in HISTORIES:
            for edge in EDGES:
                current = cell(recipient, checkpoint, history, edge)
                if metric == "axis_angle_degrees":
                    values.append(math.degrees(math.acos(np.clip(
                        current["median_within_tangent_orientation"], -1.0, 1.0
                    ))))
                elif metric == "absolute_own_minus_within":
                    values.append(abs(current["own_minus_within_median"]))
                else:
                    values.append(float(current[metric]))
    return float(np.median(values))


def body_values(
    recipients: list[dict],
    extractor: Callable[[dict], float],
) -> list[float]:
    grouped: dict[int, list[float]] = {}
    for recipient in recipients:
        grouped.setdefault(recipient["body"], []).append(float(extractor(recipient)))
    return [float(np.median(grouped[index])) for index in sorted(grouped)]


def cluster_interval(
    recipients: list[dict],
    extractor: Callable[[dict], float],
    rng: np.random.Generator,
) -> dict[str, float | int]:
    values = np.asarray(body_values(recipients, extractor), dtype=float)
    draws = rng.choice(values, size=(10_000, len(values)), replace=True)
    estimates = np.median(draws, axis=1)
    return {
        "body_clusters": len(values),
        "median_of_body_medians": float(np.median(values)),
        "cluster_ci95_low": float(np.quantile(estimates, 0.025)),
        "cluster_ci95_high": float(np.quantile(estimates, 0.975)),
        "body_q25": float(np.quantile(values, 0.25)),
        "body_q75": float(np.quantile(values, 0.75)),
        "body_minimum": float(np.min(values)),
        "body_maximum": float(np.max(values)),
    }


def diagnostic_value(recipient: dict, metric: str, amplitude: float) -> float:
    diagnostics = recipient["noise_diagnostics"]
    if metric == "applied_fine_increment_rms":
        return float(
            diagnostics.get(
                metric,
                amplitude * math.sqrt(0.01) * diagnostics["fine_step_rms"],
            )
        )
    if metric == "applied_episode_displacement_rms":
        return float(
            diagnostics.get(
                metric,
                amplitude * diagnostics["episode_displacement_rms"],
            )
        )
    return float(diagnostics[metric])


CELL_METRICS = (
    "median_within_tangent_orientation",
    "axis_angle_degrees",
    "median_within_native_tissue_orientation",
    "own_effect",
    "within_body_median",
    "own_minus_within_median",
    "absolute_own_minus_within",
    "own_percentile_within",
)

DIAGNOSTIC_METRICS = (
    "fine_step_rms",
    "applied_fine_increment_rms",
    "median_lag1_correlation",
    "spatial_effective_rank",
    "episode_displacement_effective_rank",
    "episode_displacement_rms",
    "applied_episode_displacement_rms",
)


def condition_summary(result: dict, resolution: str, rng: np.random.Generator) -> dict:
    recipients = result[resolution]
    amplitude = float(result["amplitude"])
    summary = {
        metric: cluster_interval(
            recipients,
            lambda recipient, metric=metric: recipient_metric(recipient, metric),
            rng,
        )
        for metric in CELL_METRICS
    }
    summary["noise"] = {
        metric: cluster_interval(
            recipients,
            lambda recipient, metric=metric: diagnostic_value(
                recipient, metric, amplitude
            ),
            rng,
        )
        for metric in DIAGNOSTIC_METRICS
    }
    return summary


def quantile_rmse(left: list[float], right: list[float]) -> float:
    quantiles = np.linspace(0.0, 1.0, 101)
    return float(np.sqrt(np.mean((
        np.quantile(left, quantiles) - np.quantile(right, quantiles)
    ) ** 2)))


def analyze(results: dict[str, dict]) -> dict:
    rng = np.random.default_rng(802609161)
    report: dict = {"conditions": {}, "checks": {}, "refinement": {}}
    for name, result in results.items():
        report["conditions"][name] = {
            "color": result["color"],
            "amplitude": result["amplitude"],
            "normalization": result.get("normalization", "step_rms"),
            "coarse": condition_summary(result, "coarse", rng),
            "fine": condition_summary(result, "fine", rng),
        }
        report["checks"][name] = result["checks"]
        report["refinement"][name] = {}
        for metric in CELL_METRICS:
            coarse = [recipient_metric(value, metric) for value in result["coarse"]]
            fine = [recipient_metric(value, metric) for value in result["fine"]]
            report["refinement"][name][metric] = quantile_rmse(coarse, fine)
    return report


def estimate(summary: dict) -> float:
    return float(summary["median_of_body_medians"])


def fmt(summary: dict, digits: int = 4) -> str:
    return (
        f'{summary["median_of_body_medians"]:.{digits}f} '
        f'[{summary["cluster_ci95_low"]:.{digits}f}, '
        f'{summary["cluster_ci95_high"]:.{digits}f}]'
    )


def write_summary(report: dict, destination: Path) -> None:
    lines = [
        "# O5b noise amplitude and temporal-spectrum summary",
        "",
        "Thirteen conditions use eight fixed bodies and four stochastic realizations per body. Intervals are cluster bootstraps over eight body-level medians. Every condition was run completely at dt=0.02 and dt=0.01.",
        "",
        "## Exact checks",
        "",
    ]
    for name in report["conditions"]:
        checks = report["checks"][name]
        valid = all(
            values["all_paired_initial_states_exact"]
            and values["all_realizations_share_body_template"]
            and values["all_donor_identities_valid"]
            and values["all_assay_noise_fixed"]
            and values["all_no_ac_edge"]
            and values["maximum_recipient_state_error"] == 0.0
            and values["fallback_count"] == 0
            for values in checks.values()
        )
        max_norm = max(values["maximum_norm_error"] for values in checks.values())
        max_angle = max(values["maximum_angle_error"] for values in checks.values())
        lines.append(
            f'- {name}: all structural checks {valid}; maximum norm error '
            f'{max_norm:.2e}; maximum angle error {max_angle:.2e}.'
        )

    lines.extend(["", "## White-noise amplitude series", ""])
    lines.append(
        "Values are within-body history-axis orientation, corresponding angular spread, native-tissue orientation, and absolute own-versus-donor expression-effect difference. Coarse / fine."
    )
    lines.append("")
    for name in AMPLITUDE_CONDITIONS:
        condition = report["conditions"][name]
        coarse, fine = condition["coarse"], condition["fine"]
        lines.append(
            f'- amplitude {condition["amplitude"]:.4f}: axis '
            f'{fmt(coarse["median_within_tangent_orientation"])} / '
            f'{fmt(fine["median_within_tangent_orientation"])}; angle '
            f'{fmt(coarse["axis_angle_degrees"], 2)} / '
            f'{fmt(fine["axis_angle_degrees"], 2)} degrees; native tissue '
            f'{fmt(coarse["median_within_native_tissue_orientation"])} / '
            f'{fmt(fine["median_within_native_tissue_orientation"])}; '
            f'expression gap {fmt(coarse["absolute_own_minus_within"])} / '
            f'{fmt(fine["absolute_own_minus_within"])}.'
        )

    lines.extend(["", "## Step-RMS-matched spectra", ""])
    lines.append(
        "All nonzero conditions use amplitude 0.025. Applied increment RMS is matched, while temporal correlation changes episode-scale accumulation. Coarse / fine axis results; stream diagnostics are shared across resolutions."
    )
    lines.append("")
    for name in STEP_SPECTRAL_CONDITIONS:
        condition = report["conditions"][name]
        coarse, fine = condition["coarse"], condition["fine"]
        noise = coarse["noise"]
        lines.append(
            f'- {condition["color"]}: lag-1 {fmt(noise["median_lag1_correlation"], 3)}; '
            f'spatial rank {fmt(noise["spatial_effective_rank"], 2)}; applied episode displacement '
            f'{fmt(noise["applied_episode_displacement_rms"], 4)}; axis '
            f'{fmt(coarse["median_within_tangent_orientation"])} / '
            f'{fmt(fine["median_within_tangent_orientation"])}; native tissue '
            f'{fmt(coarse["median_within_native_tissue_orientation"])} / '
            f'{fmt(fine["median_within_native_tissue_orientation"])}.'
        )

    lines.extend(["", "## Episode-displacement-matched spectra", ""])
    lines.append(
        "Mean episode-scale displacement is matched to the corresponding white stream. Per-step RMS is allowed to differ. Coarse / fine axis results."
    )
    lines.append("")
    for name in DISPLACEMENT_SPECTRAL_CONDITIONS:
        condition = report["conditions"][name]
        coarse, fine = condition["coarse"], condition["fine"]
        noise = coarse["noise"]
        lines.append(
            f'- {condition["color"]}: applied fine-increment RMS '
            f'{fmt(noise["applied_fine_increment_rms"], 5)}; applied episode displacement '
            f'{fmt(noise["applied_episode_displacement_rms"], 4)}; axis '
            f'{fmt(coarse["median_within_tangent_orientation"])} / '
            f'{fmt(fine["median_within_tangent_orientation"])}; angle '
            f'{fmt(coarse["axis_angle_degrees"], 2)} / '
            f'{fmt(fine["axis_angle_degrees"], 2)} degrees; expression gap '
            f'{fmt(coarse["absolute_own_minus_within"])} / '
            f'{fmt(fine["absolute_own_minus_within"])}.'
        )

    lines.extend(["", "## Time-step agreement", ""])
    for name in report["conditions"]:
        values = report["refinement"][name]
        lines.append(
            f'- {name}: axis-orientation quantile RMSE '
            f'{values["median_within_tangent_orientation"]:.4f}; native-tissue '
            f'{values["median_within_native_tissue_orientation"]:.4f}; expression-gap '
            f'{values["absolute_own_minus_within"]:.4f}.'
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "White perturbations span almost all 36 injected spatial dimensions, yet small white noise leaves the sedimented order axis nearly unchanged. The original O5 result therefore cannot be attributed simply to poor high-dimensional input spread. Raising white-noise amplitude progressively separates axes while native tissues remain much more closely aligned, showing that the order contrast is a sensitive geometric direction rather than a gross tissue displacement.",
        "",
        "With step RMS matched, slow and persistent spectra produce much larger axis separation than white or blue noise, but they also accumulate much greater episode-scale displacement. The post-hoc displacement-matched control retains an ordered difference: blue and white remain most aligned, OU and pink are intermediate, and brown is least aligned. Temporal organization therefore matters beyond mean episode-scale excursion in this apparatus.",
        "",
        "Despite geometric separation, the scalar magnitude of causal expression remains nearly exchangeable between the recipient's own axis and angle-matched axes from other realizations of the same body. O5b demonstrates differentiated sedimented directions, not yet recipient-unique causal-expression profiles. A later assay would need to compare the direction or composition of expression profiles rather than only their scalar distance from native expression.",
        "",
        "Neither stronger separation nor a particular spectrum is interpreted as better, resonant in Rosa's normative sense, intimate, viable, or action-worthy.",
        "",
    ])
    destination.write_text("\n".join(lines), encoding="utf-8")


def _panel_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], maximum: float, font: ImageFont.FreeTypeFont) -> None:
    left, top, right, bottom = box
    for tick in np.linspace(0.0, maximum, 5):
        y = bottom - tick / maximum * (bottom - top)
        draw.line((left, y, right, y), fill=(225, 229, 231), width=1)
        draw.text((left - 55, y - 7), f"{tick:.2f}", fill=(72, 76, 78), font=font)


def make_figure(report: dict, destination: Path) -> None:
    regular = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
    small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 12)
    title = ImageFont.truetype("C:/Windows/Fonts/seguisb.ttf", 22)
    panel_title = ImageFont.truetype("C:/Windows/Fonts/seguisb.ttf", 16)
    image = Image.new("RGB", (1600, 980), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 24), "Noise amplitude, temporal spectrum, and sedimented history-axis alignment", fill=(26, 30, 32), font=title)

    amp_box = (105, 115, 745, 415)
    _panel_axes(draw, amp_box, 1.0, small)
    amplitudes = [report["conditions"][name]["amplitude"] for name in AMPLITUDE_CONDITIONS]
    axis_values = [estimate(report["conditions"][name]["coarse"]["median_within_tangent_orientation"]) for name in AMPLITUDE_CONDITIONS]
    native_values = [estimate(report["conditions"][name]["coarse"]["median_within_native_tissue_orientation"]) for name in AMPLITUDE_CONDITIONS]
    for values, color in ((axis_values, (181, 73, 43)), (native_values, (46, 117, 111))):
        points = []
        for amplitude, value in zip(amplitudes, values):
            x = amp_box[0] + amplitude / 0.1 * (amp_box[2] - amp_box[0])
            y = amp_box[3] - value * (amp_box[3] - amp_box[1])
            points.append((x, y))
        draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    for amplitude in amplitudes:
        x = amp_box[0] + amplitude / 0.1 * (amp_box[2] - amp_box[0])
        draw.text((x - 20, amp_box[3] + 12), f"{amplitude:.3f}", fill=(55, 59, 61), font=small)
    draw.text((amp_box[0], amp_box[1] - 32), "A  White-noise amplitude series", fill=(30, 34, 36), font=panel_title)
    draw.text((amp_box[0] + 230, amp_box[3] + 42), "formation phase-noise amplitude", fill=(55, 59, 61), font=small)

    colors = ["blue", "white", "pink", "OU", "brown"]
    for panel_index, (condition_names, heading) in enumerate((
        (STEP_SPECTRAL_CONDITIONS, "Spectra matched by fine-step RMS"),
        (DISPLACEMENT_SPECTRAL_CONDITIONS, "Spectra matched by episode displacement"),
    )):
        left = 865
        top = 115 + panel_index * 395
        box = (left, top, 1510, top + 300)
        _panel_axes(draw, box, 1.0, small)
        group = (box[2] - box[0]) / len(condition_names)
        for index, name in enumerate(condition_names):
            condition = report["conditions"][name]["coarse"]
            axis = estimate(condition["median_within_tangent_orientation"])
            native = estimate(condition["median_within_native_tissue_orientation"])
            center = box[0] + group * (index + 0.5)
            for offset, value, color in (
                (-24, axis, (181, 73, 43)),
                (5, native, (46, 117, 111)),
            ):
                y = box[3] - value * (box[3] - box[1])
                draw.rectangle((center + offset, y, center + offset + 24, box[3]), fill=color)
            draw.text((center - 24, box[3] + 11), colors[index], fill=(55, 59, 61), font=small)
        panel_letter = chr(ord("B") + panel_index)
        draw.text((box[0], box[1] - 32), f"{panel_letter}  {heading}", fill=(30, 34, 36), font=panel_title)

    legend_y = 512
    draw.rectangle((105, legend_y, 126, legend_y + 15), fill=(181, 73, 43))
    draw.text((135, legend_y - 5), "history-axis orientation", fill=(45, 49, 51), font=regular)
    draw.rectangle((340, legend_y, 361, legend_y + 15), fill=(46, 117, 111))
    draw.text((370, legend_y - 5), "native-tissue orientation", fill=(45, 49, 51), font=regular)

    notes = [
        "Input spatial rank for nonzero white noise: approximately 36 of 36.",
        "Slow temporal correlation separates history axes before native tissue globally diverges.",
        "Orientation measures correspondence, not desirability or value.",
    ]
    for index, note in enumerate(notes):
        draw.text((105, 640 + index * 32), note, fill=(54, 58, 60), font=regular)
    image.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    results = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.directory.glob("*.json"))
        if path.stem in set(
            AMPLITUDE_CONDITIONS
            + STEP_SPECTRAL_CONDITIONS
            + DISPLACEMENT_SPECTRAL_CONDITIONS
        )
    }
    expected = set(
        AMPLITUDE_CONDITIONS
        + STEP_SPECTRAL_CONDITIONS
        + DISPLACEMENT_SPECTRAL_CONDITIONS
    )
    missing = expected - set(results)
    if missing:
        raise ValueError(f"missing conditions: {sorted(missing)}")
    report = analyze(results)
    (args.directory / "analysis_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(report, args.directory / "RESULTS_SUMMARY.md")
    make_figure(report, args.directory / "figure_noise_spectrum.png")
    print(args.directory / "RESULTS_SUMMARY.md")


if __name__ == "__main__":
    main()
