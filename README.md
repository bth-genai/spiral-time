# Systems article reproduction code — Spiral-Time V5

This directory is the reproduction snapshot for the four empirical result sections in he article.

1. **State-equated historical resolution:** two branches are forced to have exactly the same conventional current state while retaining different ordered encounter archives. The archive is then either hidden or exposed through an explicit multiscale history field during the same read-only continuation.
2. **Matched historical return:** intervening participation, order controls, and relational-basis controls in the finite collective-tissue apparatus.
3. **Temporal organization:** temporal-spectrum and magnitude controls in the overlapping-relations apparatus.
4. **Material-history factorial:** the complete 2 x 2 x 2 factorial for encounter order, temporal distribution, and participation.

The history-resolution extension is deliberately isolated. The shared overlapping-relations model now supports an explicit history field,

`expressed_relation = slow_tissue + beta_T * fast_trace + gamma_H * history_field`.

The state-equated experiment uses `gamma_H > 0` during the read-only continuation. The temporal-organization and material-history experiments use `gamma_H = 0` by design, so the explicit archive field does not enter their relation torque.


## V5 figure updates

V5 does not change the validated simulation equations, seeds, or numerical result files from V4. The Figure 1 generator is updated so that Panel A shows the state-equated comparison explicitly: different ordered histories are brought to the same conventional current state, then contrasted as a current-state/Markov baseline (archive hidden) and a Spiral-Time history-resolved condition (archive exposed). Panels B and C use the same stored numerical results as V4.

The Figure 3 generator also labels its three visual panels explicitly as A, B, and C so that each amplitude and temporal-spectrum comparison can be cited unambiguously in the manuscript. This presentation change does not alter the experiment, analysis, or reported values.

The Figure 4 generator applies the same explicit A, B, and C labels to tissue-axis correspondence, expression-profile correspondence, and relative non-additivity. This is likewise a presentation-only change.

## Requirements

- Python 3.12 or later
- NumPy
- pandas
- Pillow
- matplotlib

Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Verify the complete snapshot

From this directory, run:

```powershell
python run_tests.py
```

Each group is launched from its own directory so the test also verifies that dependencies resolve inside this snapshot.

## Experiment 1 — state-equated historical resolution

Primary article run (100 paired worlds; history-expression gain 0.14):

```powershell
cd state_resolution_20260920
python state_resolution_experiment.py --worlds 100 --workers 8 --history-gain 0.14 --output article_results
```

Full-archive gain sensitivity (40 paired worlds each):

```powershell
python state_resolution_experiment.py --worlds 40 --workers 8 --history-gain 0.07 --output sensitivity/g_0.07
python state_resolution_experiment.py --worlds 40 --workers 8 --history-gain 0.14 --output sensitivity/g_0.14
python state_resolution_experiment.py --worlds 40 --workers 8 --history-gain 0.28 --output sensitivity/g_0.28
```

Generate Figure 1:

```powershell
python state_resolution_figure.py --main article_results/state_resolution_results.json --sensitivity-dir sensitivity --output article_results/fig_state_resolution.png
```

The formation stage records ordered relational observations while `gamma_H = 0`. Before the probe, the blocked branch receives an exact copy of the alternating branch's conventional state: body phases, local tissues, fast traces, slow tissues, and episode index. The branch-specific encounter archives are then restored. The continuation uses identical seeds, has plasticity disabled, and does not write back to the archive. Resolution 0 hides the archive and therefore provides the exact-identity control.

## Experiment 2 — matched historical return

The article run uses 100 independently generated worlds in each of three body-capacity cells and base seed 30,510,911:

```powershell
cd experiment_1_historical_return
python collective_historical_return_experiment.py --seeds 100 --base-seed 30510911 --workers 8 --output-dir article_results
python collective_historical_return_figure.py --comparisons article_results/historical_return_comparisons.csv --trials article_results/historical_return_trials.csv --output article_results/fig_historical_return_controls.png
```

Fast structural check:

```powershell
python collective_historical_return_experiment.py --quick --seeds 2 --workers 1 --output-dir quick_check
```

## Experiment 3 — temporal organization

The article run uses eight fixed bodies, four stochastic realizations per body, and both integration steps for every frozen condition:

```powershell
cd noise_spectrum_spread_20260916
python run_all.py --bodies 8 --workers 8 --output article_results
python analyze_noise.py article_results
```

`run_all.py` is resumable; existing condition files in the selected output directory are not recalculated.

## Experiment 4 — material-history factorial

The article run uses the complete 2 x 2 x 2 factorial for eight fixed bodies, four realizations per body, and both integration steps:

```powershell
cd material_history_factorial_20260916
python material_experiment.py --bodies 8 --workers 8 --output article_results
python analyze_material.py article_results/results.json
```

## Directory map

- `state_resolution_20260920/`: state-equated experiment, figure generator, tests, primary data, and gain-sensitivity data.
- `experiment_1_historical_return/`: collective exact-return experiment, controls, figure code, and tests.
- `noise_spectrum_spread_20260916/`: temporal-organization runner, analysis, figures, and tests.
- `material_history_factorial_20260916/`: material-history factorial, analysis, figures, and tests.
- `overlapping_relations_20260915/`: shared oscillator model, now including explicit ordered encounter archives and multiscale history-field feedback.
- `geometry_specificity_20260915/`, `geometry_directionality_20260915/`, `history_axis_specificity_20260915/`: shared measurement and control dependencies.

The package does not require the surrounding project tree at runtime.
