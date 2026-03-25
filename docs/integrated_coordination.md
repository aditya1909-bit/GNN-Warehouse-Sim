# Integrated Coordination

## Purpose

Stage 12 adds a second coordination stack beside the dispatch-centric simulator.

This stack supports:

- centralized multi-robot coordination
- continuous-time motion on the warehouse graph
- prioritized SIPP-style planning with timed trajectories
- exact current-epoch joint route search over the selected macro candidate set
- explicit collision-event reporting
- an experimental end-to-end macro PPO controller

## Coordination Mode

Integrated runs use:

```toml
[simulation]
coordination_mode = "integrated"
policy = "prioritized_sipp_coordinator"
execution_model = "idealized"

[coordination]
control_dt = 0.25
replan_period = 1.0
robot_radius = 0.2
collision_clearance = 0.05
k_shortest_paths = 3
max_route_options_per_pair = 3
```

`execution_model` stays `idealized` in config for compatibility, but integrated mode ignores the old reservation execution stack and always uses continuous graph execution internally.

## Implemented Policies

- `prioritized_sipp_coordinator`: centralized non-learning baseline
- `optimal_mapf_coordinator`: exact joint-search router over the current replan epoch's assigned macro set
- `random_macro`: weak integrated smoke baseline
- `trained_end_to_end_macro_ppo`: artifact-backed macro PPO controller

The `optimal_mapf_coordinator` claim is intentionally bounded. It is exact over the current epoch's finite task-route macro candidates and continuous graph-execution realization, not over future task releases or the full warehouse-level task-allocation problem.

## Outputs

Integrated runs write the standard summary files plus:

- `robot_trajectories.csv`
- `macro_decisions.csv`
- `collision_events.csv`
- `planner_plans.csv`

Summary metrics now also include:

- `safety_violations_total`
- `replans_total`
- `planner_failures_total`

## Claim Boundary

The integrated learned controller exists and can be trained and loaded back into the simulator.

The repository should only claim learned end-to-end warehouse coordination when held-out repeated-seed benchmarks satisfy the configured benchmark gate:

- zero safety violations
- task completion rate threshold
- throughput ratio threshold versus `prioritized_sipp_coordinator`

Until then, the learned policy should be described as an experimental integrated coordinator.
