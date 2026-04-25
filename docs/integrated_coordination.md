# Integrated Coordination

## Purpose

Stage 12 adds a second coordination stack beside the dispatch-centric simulator.

This stack supports:

- centralized multi-robot coordination
- graph-embedded continuous-time motion on the warehouse graph
- continuous off-graph motion between node coordinates with polygon obstacle avoidance
- prioritized SIPP-style planning with timed trajectories
- exact current-epoch joint route search over the selected macro candidate set
- multi-route continuous branching over per-leg geometric alternatives
- battery-aware charge macros alongside task macros
- explicit collision-event reporting
- a conflict-aware dense-traffic macro PPO controller

## Coordination Mode

Integrated runs use:

```toml
[simulation]
coordination_mode = "integrated"
policy = "prioritized_sipp_coordinator"
execution_model = "idealized"

[coordination]
motion_model = "graph_embedded"
control_dt = 0.25
replan_period = 1.0
robot_radius = 0.2
collision_clearance = 0.05
k_shortest_paths = 3
max_route_options_per_pair = 3
```

`execution_model` stays `idealized` in config for compatibility, but integrated mode ignores the old reservation execution stack and always uses continuous motion internally. `coordination.motion_model = "graph_embedded"` preserves the existing edge-constrained realization. `coordination.motion_model = "free_space"` and `coordination.motion_model = "obstacle_aware_free_space"` now resolve through the same continuous planner; the former remains as a compatibility alias while the planner consumes polygon obstacles when present.

## Implemented Policies

- `prioritized_sipp_coordinator`: centralized non-learning baseline
- `optimal_mapf_coordinator`: exact joint-search router over the current replan epoch's assigned macro set
- `random_macro`: weak integrated smoke baseline
- `trained_conflict_graph_macro_ppo`: artifact-backed conflict-graph macro PPO controller
- `trained_end_to_end_macro_ppo`: backward-compatible legacy policy name for older artifacts/configs

The `optimal_mapf_coordinator` claim is intentionally bounded. It is exact over the current epoch's finite macro candidate surface and realized continuous trajectories, not over future task releases or the full warehouse-level task-allocation problem.

The continuous motion mode is also intentionally bounded. It now supports explicit polygon obstacles, blocked-cell square obstacles, and multiple geometric alternatives per leg, but it is still not a full rigid-body physics or warehouse-CAD stack.

## Outputs

Integrated runs write the standard summary files plus:

- `robot_trajectories.csv`
- `macro_decisions.csv`
- `collision_events.csv`
- `planner_plans.csv`
- `charging_executions.csv`

Summary metrics now also include:

- `safety_violations_total`
- `replans_total`
- `planner_failures_total`
- `total_energy_consumed`
- `total_energy_charged`
- `total_charging_time`

## Claim Boundary

The integrated learned controller exists and can be trained and loaded back into the simulator.

The repository should only claim dense-traffic learned warehouse coordination when held-out repeated-seed benchmarks satisfy the configured benchmark gate:

- zero safety violations
- task completion rate threshold
- throughput ratio threshold versus `prioritized_sipp_coordinator`
- policy-distinctness threshold versus the warm-start teacher

Until then, the learned policy should be described as a benchmark-gated dense-traffic integrated coordinator.
