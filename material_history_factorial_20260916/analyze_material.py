"""Body-clustered analysis and figures for O6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHECKPOINTS = ("formation", "withdrawal", "return_6")
EDGES = ("AB", "BC")
FACTORS = ("order", "timing", "participation")
INTERACTIONS = (
    "order_x_timing",
    "order_x_participation",
    "timing_x_participation",
)
PROFILE_COMPONENTS = ("body_A", "body_B", "body_C", "alignment_AB", "alignment_BC")


def body_values(
    recipients: list[dict], extractor: Callable[[dict], float]
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
    medians = np.median(draws, axis=1)
    return {
        "body_clusters": len(values),
        "median_of_body_medians": float(np.median(values)),
        "cluster_ci95_low": float(np.quantile(medians, 0.025)),
        "cluster_ci95_high": float(np.quantile(medians, 0.975)),
        "body_q25": float(np.quantile(values, 0.25)),
        "body_q75": float(np.quantile(values, 0.75)),
        "body_minimum": float(np.min(values)),
        "body_maximum": float(np.max(values)),
    }


def factor_values(
    recipient: dict,
    factor: str,
    metric: str,
    checkpoint: str | None = None,
) -> list[float]:
    checkpoints = (checkpoint,) if checkpoint else CHECKPOINTS
    output = []
    for current_checkpoint in checkpoints:
        for edge in EDGES:
            value = recipient["cells"][current_checkpoint][edge]["factors"][factor]
            if metric in (
                "median_tissue_axis_orientation",
                "median_profile_axis_orientation",
                "median_expression_effect",
            ):
                output.append(float(value[metric]))
            elif metric == "median_tissue_angle_degrees":
                output.extend(
                    float(item["tissue_angle_degrees"])
                    for item in value["contrasts"]
                )
            elif metric == "median_raw_norm_ratio":
                output.extend(
                    float(item["target_to_native_norm_ratio"])
                    for item in value["contrasts"]
                )
            elif metric.startswith("absolute_component_"):
                component = metric.removeprefix("absolute_component_")
                output.extend(
                    abs(float(item["expression_profile_components"][component]))
                    for item in value["contrasts"]
                )
            else:
                raise KeyError(metric)
    return output


def factor_interval(
    recipients: list[dict],
    factor: str,
    metric: str,
    checkpoint: str | None,
    rng: np.random.Generator,
) -> dict:
    return cluster_interval(
        recipients,
        lambda recipient: float(np.median(
            factor_values(recipient, factor, metric, checkpoint)
        )),
        rng,
    )


FACTOR_METRICS = (
    "median_tissue_axis_orientation",
    "median_profile_axis_orientation",
    "median_expression_effect",
    "median_tissue_angle_degrees",
    "median_raw_norm_ratio",
)


def interaction_values(
    recipient: dict,
    interaction: str,
    checkpoint: str | None = None,
) -> list[float]:
    checkpoints = (checkpoint,) if checkpoint else CHECKPOINTS
    return [
        float(item["relative_residual"])
        for current_checkpoint in checkpoints
        for edge in EDGES
        for item in recipient["cells"][current_checkpoint][edge]["interactions"][interaction]
    ]


def resolution_summary(recipients: list[dict], rng: np.random.Generator) -> dict:
    summary: dict = {"pooled": {"factors": {}, "interactions": {}}, "checkpoints": {}}
    for factor in FACTORS:
        summary["pooled"]["factors"][factor] = {
            metric: factor_interval(recipients, factor, metric, None, rng)
            for metric in FACTOR_METRICS
        }
        summary["pooled"]["factors"][factor]["profile_components"] = {
            component: factor_interval(
                recipients,
                factor,
                f"absolute_component_{component}",
                None,
                rng,
            )
            for component in PROFILE_COMPONENTS
        }
    for interaction in INTERACTIONS:
        summary["pooled"]["interactions"][interaction] = cluster_interval(
            recipients,
            lambda recipient, interaction=interaction: float(np.median(
                interaction_values(recipient, interaction)
            )),
            rng,
        )
    for checkpoint in CHECKPOINTS:
        summary["checkpoints"][checkpoint] = {"factors": {}, "interactions": {}}
        for factor in FACTORS:
            summary["checkpoints"][checkpoint]["factors"][factor] = {
                metric: factor_interval(
                    recipients, factor, metric, checkpoint, rng
                )
                for metric in FACTOR_METRICS
            }
        for interaction in INTERACTIONS:
            summary["checkpoints"][checkpoint]["interactions"][interaction] = cluster_interval(
                recipients,
                lambda recipient, interaction=interaction, checkpoint=checkpoint: float(
                    np.median(interaction_values(recipient, interaction, checkpoint))
                ),
                rng,
            )
    return summary


def quantile_rmse(left: list[float], right: list[float]) -> float:
    quantiles = np.linspace(0.0, 1.0, 101)
    return float(np.sqrt(np.mean((
        np.quantile(left, quantiles) - np.quantile(right, quantiles)
    ) ** 2)))


def analyze(result: dict) -> dict:
    rng = np.random.default_rng(902609161)
    report = {
        "body_count": result["body_count"],
        "realizations_per_body": result["realizations_per_body"],
        "checks": result["checks"],
        "coarse": resolution_summary(result["coarse"], rng),
        "fine": resolution_summary(result["fine"], rng),
        "refinement": {"factors": {}, "interactions": {}},
    }
    for factor in FACTORS:
        report["refinement"]["factors"][factor] = {}
        for metric in FACTOR_METRICS:
            coarse = [
                float(np.median(factor_values(value, factor, metric)))
                for value in result["coarse"]
            ]
            fine = [
                float(np.median(factor_values(value, factor, metric)))
                for value in result["fine"]
            ]
            report["refinement"]["factors"][factor][metric] = quantile_rmse(
                coarse, fine
            )
    for interaction in INTERACTIONS:
        coarse = [
            float(np.median(interaction_values(value, interaction)))
            for value in result["coarse"]
        ]
        fine = [
            float(np.median(interaction_values(value, interaction)))
            for value in result["fine"]
        ]
        report["refinement"]["interactions"][interaction] = quantile_rmse(
            coarse, fine
        )
    return report


def fmt(value: dict, digits: int = 3) -> str:
    return (
        f'{value["median_of_body_medians"]:.{digits}f} '
        f'[{value["cluster_ci95_low"]:.{digits}f}, '
        f'{value["cluster_ci95_high"]:.{digits}f}]'
    )


def write_summary(report: dict, destination: Path) -> None:
    lines = [
        "# O6 material-history factorial summary",
        "",
        f'Based on {report["body_count"]} fixed bodies, '
        f'{report["realizations_per_body"]} shared-noise realizations per body, '
        "eight material histories, three checkpoints, and complete coarse/fine runs.",
        "",
        "Intervals are cluster bootstraps over eight body-level medians.",
        "",
        "## Exact checks",
        "",
    ]
    for resolution in ("coarse", "fine"):
        checks = report["checks"][resolution]
        lines.append(
            f'- {resolution}: exact starts {checks["all_initial_states_exact"]}; '
            f'no AC storage {checks["all_no_ac_edge"]}; write-back error '
            f'{checks["maximum_recipient_state_error"]:.2e}; norm-match error '
            f'{checks["maximum_norm_match_error"]:.2e}; undefined tissue/profile '
            f'axis pairs {checks["undefined_tissue_axis_pairs"]}/'
            f'{checks["undefined_profile_axis_pairs"]}.'
        )

    lines.extend(["", "## Pooled material-history factors", ""])
    lines.append(
        "Values are coarse / fine. Axis orientation describes correspondence of one factor's direction across the four contexts created by the other two factors."
    )
    lines.append("")
    for factor in FACTORS:
        coarse = report["coarse"]["pooled"]["factors"][factor]
        fine = report["fine"]["pooled"]["factors"][factor]
        lines.append(
            f'- {factor}: tissue-axis orientation '
            f'{fmt(coarse["median_tissue_axis_orientation"])} / '
            f'{fmt(fine["median_tissue_axis_orientation"])}; profile-axis orientation '
            f'{fmt(coarse["median_profile_axis_orientation"])} / '
            f'{fmt(fine["median_profile_axis_orientation"])}; tissue angle '
            f'{fmt(coarse["median_tissue_angle_degrees"], 2)} / '
            f'{fmt(fine["median_tissue_angle_degrees"], 2)} degrees; expression effect '
            f'{fmt(coarse["median_expression_effect"], 4)} / '
            f'{fmt(fine["median_expression_effect"], 4)}; raw target/native norm ratio '
            f'{fmt(coarse["median_raw_norm_ratio"])} / '
            f'{fmt(fine["median_raw_norm_ratio"])}.'
        )

    lines.extend(["", "## Persistence across checkpoints", ""])
    for checkpoint in CHECKPOINTS:
        lines.append(f'### {checkpoint.replace("_", " ").title()}')
        lines.append("")
        for factor in FACTORS:
            coarse = report["coarse"]["checkpoints"][checkpoint]["factors"][factor]
            fine = report["fine"]["checkpoints"][checkpoint]["factors"][factor]
            lines.append(
                f'- {factor}: tissue-axis {fmt(coarse["median_tissue_axis_orientation"])} / '
                f'{fmt(fine["median_tissue_axis_orientation"])}; profile-axis '
                f'{fmt(coarse["median_profile_axis_orientation"])} / '
                f'{fmt(fine["median_profile_axis_orientation"])}; effect '
                f'{fmt(coarse["median_expression_effect"], 4)} / '
                f'{fmt(fine["median_expression_effect"], 4)}.'
            )
        lines.append("")

    lines.extend(["## Pairwise non-additivity", ""])
    for interaction in INTERACTIONS:
        coarse = report["coarse"]["pooled"]["interactions"][interaction]
        fine = report["fine"]["pooled"]["interactions"][interaction]
        lines.append(
            f'- {interaction}: relative parallelogram residual '
            f'{fmt(coarse)} / {fmt(fine)}.'
        )

    lines.extend(["", "## Expression-profile composition", ""])
    lines.append(
        "Median absolute component changes are listed in A, B, C, AB, BC order. They are descriptive and retain which parts of the assay express a factor."
    )
    lines.append("")
    for factor in FACTORS:
        coarse = report["coarse"]["pooled"]["factors"][factor]["profile_components"]
        fine = report["fine"]["pooled"]["factors"][factor]["profile_components"]
        coarse_values = ", ".join(
            f'{coarse[value]["median_of_body_medians"]:.4f}' for value in PROFILE_COMPONENTS
        )
        fine_values = ", ".join(
            f'{fine[value]["median_of_body_medians"]:.4f}' for value in PROFILE_COMPONENTS
        )
        lines.append(f'- {factor}: {coarse_values} / {fine_values}.')

    lines.extend(["", "## Time-step comparison", ""])
    for factor in FACTORS:
        values = report["refinement"]["factors"][factor]
        lines.append(
            f'- {factor}: tissue-axis quantile RMSE '
            f'{values["median_tissue_axis_orientation"]:.4f}; profile-axis '
            f'{values["median_profile_axis_orientation"]:.4f}; expression effect '
            f'{values["median_expression_effect"]:.4f}.'
        )
    for interaction, value in report["refinement"]["interactions"].items():
        lines.append(f'- {interaction}: residual quantile RMSE {value:.4f}.')

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Order, temporal distribution, and participation all create persistent tissue displacements and causal expression changes. Their scalar expression magnitudes overlap substantially, but the geometric and expression-profile directions have low to moderate correspondence across contexts. The same named history factor therefore does not denote one context-independent transformation.",
        "",
        "Relative parallelogram residuals are of the same order as the adjacent main-effect displacements. The material-history cube is strongly non-additive: the geometry produced by changing two factors cannot be reconstructed by summing two fixed factor vectors. This is a classical dynamical interaction result, not a quantum claim.",
        "",
        "Participation is especially context-dependent in expression-profile direction. Reducing encounter count does not enact one uniform loss; what changes in A, B, C, AB, and BC depends on order and temporal distribution. The retained full profile reveals a distinction that the scalar effect magnitude alone obscures.",
        "",
        "No factor level or magnitude is interpreted as better, more viable, intimate, resonant, or preferable.",
        "",
    ])
    destination.write_text("\n".join(lines), encoding="utf-8")


def estimate(value: dict) -> float:
    return float(value["median_of_body_medians"])


def make_figure(report: dict, destination: Path) -> None:
    regular = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
    small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 12)
    title = ImageFont.truetype("C:/Windows/Fonts/seguisb.ttf", 22)
    panel_title = ImageFont.truetype("C:/Windows/Fonts/seguisb.ttf", 16)
    image = Image.new("RGB", (1550, 860), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 24), "Material histories enact context-dependent geometry and expression", fill=(25, 29, 31), font=title)
    panels = (
        ("median_tissue_axis_orientation", "Tissue-axis correspondence", 1.0),
        ("median_profile_axis_orientation", "Expression-profile correspondence", 1.0),
    )
    colors = {"order": (181, 73, 43), "timing": (46, 117, 111), "participation": (77, 91, 117)}
    for panel_index, (metric, heading, maximum) in enumerate(panels):
        left = 95 + panel_index * 755
        top, width, height = 125, 620, 270
        for tick in np.linspace(-0.25, maximum, 6):
            y = top + height - (tick + 0.25) / 1.25 * height
            draw.line((left, y, left + width, y), fill=(225, 229, 231), width=1)
            draw.text((left - 52, y - 7), f"{tick:.2f}", fill=(70, 74, 76), font=small)
        group = width / len(CHECKPOINTS)
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            center = left + group * (checkpoint_index + 0.5)
            for factor_index, factor in enumerate(FACTORS):
                value = estimate(
                    report["coarse"]["checkpoints"][checkpoint]["factors"][factor][metric]
                )
                x = center - 48 + factor_index * 34
                zero_y = top + height - 0.25 / 1.25 * height
                y = top + height - (value + 0.25) / 1.25 * height
                draw.rectangle((x, min(y, zero_y), x + 28, max(y, zero_y)), fill=colors[factor])
            draw.text((center - 36, top + height + 12), checkpoint.replace("_", " "), fill=(52, 56, 58), font=small)
        panel_letter = chr(ord("A") + panel_index)
        draw.text((left, top - 34), f"{panel_letter}  {heading}", fill=(30, 34, 36), font=panel_title)

    left, top, width, height = 270, 540, 1010, 220
    maximum = 1.5
    for tick in np.linspace(0.0, maximum, 4):
        y = top + height - tick / maximum * height
        draw.line((left, y, left + width, y), fill=(225, 229, 231), width=1)
        draw.text((left - 52, y - 7), f"{tick:.1f}", fill=(70, 74, 76), font=small)
    interaction_colors = ((181, 73, 43), (46, 117, 111), (77, 91, 117))
    group = width / len(CHECKPOINTS)
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        center = left + group * (checkpoint_index + 0.5)
        for index, interaction in enumerate(INTERACTIONS):
            value = estimate(
                report["coarse"]["checkpoints"][checkpoint]["interactions"][interaction]
            )
            x = center - 51 + index * 36
            y = top + height - value / maximum * height
            draw.rectangle((x, y, x + 30, top + height), fill=interaction_colors[index])
        draw.text((center - 36, top + height + 12), checkpoint.replace("_", " "), fill=(52, 56, 58), font=small)
    draw.text((left, top - 34), "C  Relative non-additivity", fill=(30, 34, 36), font=panel_title)

    legend_x = 485
    for index, factor in enumerate(FACTORS):
        draw.rectangle((legend_x, 445, legend_x + 20, 460), fill=colors[factor])
        draw.text((legend_x + 28, 439), factor, fill=(44, 48, 50), font=regular)
        legend_x += 175
    image.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    report = analyze(result)
    destination = args.results.parent
    (destination / "analysis_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(report, destination / "RESULTS_SUMMARY.md")
    make_figure(report, destination / "figure_material_history.png")
    print(destination / "RESULTS_SUMMARY.md")


if __name__ == "__main__":
    main()
