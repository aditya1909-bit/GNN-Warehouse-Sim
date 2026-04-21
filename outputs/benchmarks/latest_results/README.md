# Latest Heavy Results

This folder packages the latest completed heavy benchmark outputs that are ready to upload.

Included bundles:

- `dispatch_heavy/`: copied from `outputs/benchmarks/canonical_full_matrix_heavy/dispatch`
- `integrated_heavy/`: copied from `outputs/benchmarks/canonical_integrated_benchmark_heavy`

Important status note:

- The full heavy matrix under `outputs/benchmarks/canonical_full_matrix_heavy` is not fully complete because its integrated half is still partial.
- This upload bundle therefore uses the completed heavy dispatch benchmark and the separately completed heavy integrated benchmark.

## Main Takeaways

### Dispatch Heavy

The heavy dispatch suite is mostly a tie outside the due-time-pressure case.

- `dispatch_due_pressure_heavy` is the informative scenario.
- `congestion_aware_nearest_robot_task` is the strongest dispatch policy there.
- It improves throughput to about `446.6`, versus about `421.5` for `nearest_robot_task` and `420.4` for `fifo`.
- It also has the best due-time behavior:
  - `on_time_completion_rate` about `0.71`
  - `overdue_task_count` about `91`
- The learned dispatch policies do not win the heavy dispatch suite.
- `trained_mlp_model` is the weakest learned dispatch model in the due-pressure regime.

For the other heavy dispatch scenarios:

- `open_high_load`, `dense_crossing_heavy`, and `high_fleet_density` are effectively collapse/tie regimes.
- The learned models and the stronger heuristics often land on the same aggregate outcomes.

### Integrated Heavy

The heavy integrated suite supports a planner-first story.

- `integrated_narrow_bottleneck` remains the cleanest planner result.
- `optimal_mapf_coordinator` has the best raw p95 completion time there at about `38.2`.
- `prioritized_sipp_coordinator` is still clearly better than `random_macro`.

- `integrated_tight_chokepoint_heavy` is the clearest hard-case separation.
- Throughput is basically tied across policies, but collision count is not:
  - `prioritized_sipp_coordinator`: about `8.16`
  - `optimal_mapf_coordinator`: about `11.64`
  - `random_macro`: about `18.84`
  - `trained_end_to_end_macro_ppo`: about `23.52`

- `integrated_high_fleet_density_heavy` is the one heavy integrated scenario where the learned macro controller looks promising.
- Collision counts there are:
  - `trained_end_to_end_macro_ppo`: about `2.44`
  - `prioritized_sipp_coordinator`: about `3.16`
  - `optimal_mapf_coordinator`: about `4.20`
  - `random_macro`: about `7.44`
- That is interesting, but not enough yet to support a broad “learned integrated wins” claim.

- `integrated_free_space` is mostly a low-contention tie regime.

## Recommended Narrative

The most defensible current story is:

- heavy dispatch results validate the due-time-pressure scenario, but the best policy is still a congestion-aware heuristic
- heavy integrated results strengthen the planner-first claim under contention
- learned integrated control is mixed: promising in high fleet density, but still collapsed or worse in other regimes

## Suggested Figures

Dispatch:

- `dispatch_heavy/figures/claim_forest_plot.png`
- `dispatch_heavy/figures/paired_seed_dot_plot.png`
- `dispatch_heavy/figures/dispatch_decision_explainer.png`

Integrated:

- `integrated_heavy/figures/claim_forest_plot.png`
- `integrated_heavy/figures/integrated_narrow_bottleneck_mechanism.png`
- `integrated_heavy/figures/integrated_narrow_bottleneck_congestion_heatmap.png`
- `integrated_heavy/figures/policy_collapse_diagnostics.png`
