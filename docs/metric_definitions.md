# Metric Definitions

The benchmark layer uses one canonical schema across dispatch, integrated coordination, planner, and learning outputs.

## Primary Coordination Metrics

- `throughput`: completed tasks per hour over realized makespan.
- `mean_task_completion_time`: mean task turnaround time from release to completion.
- `p95_task_completion_time`: 95th percentile turnaround time.
- `makespan`: total elapsed simulation time until realized work completes.
- `mean_queue_length`: time-weighted mean ready-task queue length.
- `p95_queue_length`: 95th percentile ready-task queue length across event snapshots.
- `robot_idle_fraction`: fleet-level idle-time share over realized makespan.
- `travel_distance_per_completed_task`: mean realized robot travel distance per completed task.
- `realized_waiting_time`: total execution-time waiting inserted by congestion or reservations.
- `congestion_event_count`: total blocked-traversal or congestion-induced wait events.
- `collision_event_count`: explicit safety-violation or collision events.
- `deadlock_livelock_incident_count`: explicit deadlock or livelock incidents. The current stack emits `0` unless a detector writes incidents.
- `on_time_completion_rate`: fraction of completed tasks that finish on or before their due time.
- `mean_tardiness`: mean `max(completion_time - due_time, 0)` over completed tasks.
- `p95_tardiness`: 95th percentile tardiness over completed tasks.
- `overdue_task_count`: count of completed tasks that miss their due time.

## Planner Metrics

- `planning_latency`: mean wall-clock planner latency per planner call. The current stack reserves this field but does not yet emit live latency measurements.
- `replanning_count`: number of distinct replanning epochs in the run.
- `planner_failure_count`: planner calls that failed to produce a feasible plan.
- `timeout_count`: planner calls that timed out.
- `path_conflict_count_before_resolution`: raw conflicts detected before planner conflict resolution, aggregated once per planning epoch.
- `sipp_wait_insertion_count`: explicit SIPP wait insertions attributable to conflict avoidance.
- `planner_wait_time_total`: total wait time inserted by the planner to resolve motion conflicts.
- `mapf_solve_success_rate`: fraction of planner calls that produced a feasible joint plan.

## Learning Metrics

- `reward_mean`
- `reward_std`
- `policy_entropy`
- `invalid_action_rate`
- `masked_action_rejection_rate`
- `ppo_kl`
- `ppo_clip_fraction`
- `value_loss`
- `generalization_gap_seen_vs_unseen`

Those fields are part of the stable schema now so offline fitting, PPO fine-tuning, and macro-controller studies can write comparable machine-readable outputs as the ablation work lands.

## Notes

- `metric_schema_version` is written into benchmark summaries, aggregates, and claims.
- Aggregate tables append `_mean`, `_std`, `_ci95_low`, and `_ci95_high` to every canonical metric name.
- When a metric is defined in the schema but not yet emitted by the underlying subsystem, the output writes `null` in JSON and an empty CSV cell.
- `METRIC_SCHEMA_VERSION` is now `1.1`, which is the first schema version that requires the due-time and planner-conflict fields above.
