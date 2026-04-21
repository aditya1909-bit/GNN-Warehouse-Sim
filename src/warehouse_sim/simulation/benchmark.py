"""Benchmark runner for comparing policies across scenarios."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
from math import ceil
import os
from pathlib import Path
import time

from warehouse_sim.config import BenchmarkConfig, ExperimentConfig, load_benchmark_config, load_experiment_config
from warehouse_sim.config.models import PolicyModelConfig
from warehouse_sim.metrics import write_benchmark_report
from warehouse_sim.reporting import METRIC_SCHEMA_VERSION, build_simulation_metric_record, load_artifact_aliases
from warehouse_sim.simulation.runner import default_parallel_worker_count, resolve_runtime_device, run_experiment_from_config

_ARTIFACT_FILENAMES: dict[str, str] = {
    "summary": "summary.json",
    "executions": "executions.csv",
    "dispatch_traces": "dispatch_traces.csv",
    "dispatch_node_observations": "dispatch_node_observations.csv",
    "dispatch_arc_observations": "dispatch_arc_observations.csv",
    "queue_snapshots": "queue_snapshots.csv",
    "robot_metrics": "robot_metrics.csv",
    "charging_executions": "charging_executions.csv",
    "robot_trajectories": "robot_trajectories.csv",
    "macro_decisions": "macro_decisions.csv",
    "collision_events": "collision_events.csv",
    "planner_plans": "planner_plans.csv",
}


@dataclass(frozen=True)
class BenchmarkJob:
    """One independent benchmark run."""

    benchmark_name: str
    scenario_family: str
    scenario_id: str
    scenario_name: str
    scenario_config: Path
    seed: int
    policy: str
    policy_family: str
    policy_role: str
    coordination_mode: str
    execution_model: str
    motion_model: str
    fleet_size: int
    demand_mean_interval: float
    demand_horizon_seconds: float
    layout_rows: int
    layout_columns: int
    blocked_cell_count: int
    directed_edge_count: int
    topology_difficulty: str
    run_output_dir: Path
    force_write_plots: bool
    policy_artifact_path: Path | None = None
    runtime_device: str = "cpu"

    @property
    def job_id(self) -> str:
        return f"{self.scenario_id}::seed_{self.seed}::{self.policy}"


def run_benchmark_from_config(
    benchmark_config: BenchmarkConfig,
    benchmark_root_override: Path | None = None,
    force_write_plots: bool | None = None,
    parallel_workers_override: int | None = None,
    resume_override: bool | None = None,
    fail_fast_override: bool | None = None,
    use_mps_for_learned_policies_override: bool | None = None,
) -> dict[str, Path]:
    """Run a policy benchmark from a loaded benchmark config."""

    benchmark_root = benchmark_root_override or benchmark_config.output_dir
    benchmark_root.mkdir(parents=True, exist_ok=True)
    effective_config = replace(
        benchmark_config,
        output_dir=benchmark_root,
        parallel_workers=parallel_workers_override if parallel_workers_override is not None else benchmark_config.parallel_workers,
        resume=benchmark_config.resume if resume_override is None else resume_override,
        fail_fast=benchmark_config.fail_fast if fail_fast_override is None else fail_fast_override,
        use_mps_for_learned_policies=(
            benchmark_config.use_mps_for_learned_policies
            if use_mps_for_learned_policies_override is None
            else use_mps_for_learned_policies_override
        ),
    )
    effective_force_write_plots = effective_config.write_plots if force_write_plots is None else force_write_plots
    run_state_path = benchmark_root / "run_state.json"

    jobs, scenario_seed_map, config_sources = _build_benchmark_jobs(
        effective_config,
        benchmark_root=benchmark_root,
        force_write_plots=effective_force_write_plots,
    )
    total_jobs = len(jobs)
    summary_rows: list[dict[str, object]] = []
    state = _initial_run_state(effective_config.name, total_jobs)
    pending_jobs: list[BenchmarkJob] = []
    resumed_count = 0

    for job in jobs:
        if effective_config.resume and _is_completed_run(job.run_output_dir):
            summary_rows.append(_load_completed_row(job))
            state["jobs"][job.job_id] = {"status": "skipped", "output_dir": str(job.run_output_dir)}
            resumed_count += 1
        else:
            pending_jobs.append(job)
            state["jobs"][job.job_id] = {"status": "queued", "output_dir": str(job.run_output_dir)}

    state["counts"]["queued"] = len(pending_jobs)
    state["counts"]["skipped"] = resumed_count
    _write_run_state(run_state_path, state)

    failed_jobs: list[dict[str, str]] = []
    if pending_jobs:
        worker_count = _effective_parallel_workers(effective_config.parallel_workers)
        batch_rows, failed_jobs = _run_pending_jobs(
            pending_jobs,
            total_jobs=total_jobs,
            starting_completed=len(summary_rows),
            worker_count=worker_count,
            fail_fast=effective_config.fail_fast,
            run_state_path=run_state_path,
            state=state,
        )
        summary_rows.extend(batch_rows)

    execution_summary_path = benchmark_root / "execution_summary.json"
    execution_summary = {
        "benchmark_name": effective_config.name,
        "parallel_workers": _effective_parallel_workers(effective_config.parallel_workers),
        "resume": effective_config.resume,
        "fail_fast": effective_config.fail_fast,
        "use_mps_for_learned_policies": effective_config.use_mps_for_learned_policies,
        "completed_jobs": len(summary_rows),
        "skipped_jobs": resumed_count,
        "failed_jobs": failed_jobs,
        "total_jobs": total_jobs,
    }
    execution_summary_path.write_text(json.dumps(execution_summary, indent=2), encoding="utf-8")

    if failed_jobs:
        raise RuntimeError(
            f"Benchmark {effective_config.name} failed for {len(failed_jobs)} job(s); "
            f"see {run_state_path} for details."
        )

    summary_rows.sort(key=_row_sort_key)
    aggregate_paths = write_benchmark_report(
        output_dir=benchmark_root,
        benchmark_name=effective_config.name,
        rows=summary_rows,
        config_sources={
            "benchmark": _benchmark_config_snapshot(effective_config),
            **config_sources,
        },
        seed_bundle={
            "benchmark_name": effective_config.name,
            "scenario_family": effective_config.scenario_family,
            "shared_across_policies": True,
            "scenario_seeds": scenario_seed_map,
        },
        write_manifest=effective_config.write_manifest,
    )
    aggregate_paths["run_state"] = run_state_path
    aggregate_paths["execution_summary"] = execution_summary_path
    return aggregate_paths


def run_benchmark_from_path(
    benchmark_config_path: Path,
    benchmark_root_override: Path | None = None,
    force_write_plots: bool | None = None,
    parallel_workers_override: int | None = None,
    resume_override: bool | None = None,
    fail_fast_override: bool | None = None,
    use_mps_for_learned_policies_override: bool | None = None,
    scenario_filters: tuple[str, ...] | None = None,
    policy_filters: tuple[str, ...] | None = None,
    seed_filters: tuple[int, ...] | None = None,
) -> dict[str, Path]:
    """Load a benchmark config and run it."""

    benchmark_config = load_benchmark_config(benchmark_config_path)
    benchmark_config = _resolve_benchmark_paths(benchmark_config, benchmark_config_path.parent)
    benchmark_config = _filter_benchmark_config(
        benchmark_config,
        scenario_filters=scenario_filters,
        policy_filters=policy_filters,
        seed_filters=seed_filters,
    )
    return run_benchmark_from_config(
        benchmark_config=benchmark_config,
        benchmark_root_override=benchmark_root_override,
        force_write_plots=force_write_plots,
        parallel_workers_override=parallel_workers_override,
        resume_override=resume_override,
        fail_fast_override=fail_fast_override,
        use_mps_for_learned_policies_override=use_mps_for_learned_policies_override,
    )


def _run_pending_jobs(
    jobs: list[BenchmarkJob],
    *,
    total_jobs: int,
    starting_completed: int,
    worker_count: int,
    fail_fast: bool,
    run_state_path: Path,
    state: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    rows: list[dict[str, object]] = []
    failed_jobs: list[dict[str, str]] = []
    mps_jobs = [job for job in jobs if job.runtime_device == "mps"]
    cpu_jobs = [job for job in jobs if job.runtime_device != "mps"]
    completion_count = starting_completed

    if worker_count <= 1 or not mps_jobs or not cpu_jobs:
        result_rows, failures, completion_count = _run_job_batch(
            jobs,
            max_workers=max(1, worker_count),
            total_jobs=total_jobs,
            starting_completed=starting_completed,
            run_state_path=run_state_path,
            state=state,
            fail_fast=fail_fast,
        )
        rows.extend(result_rows)
        failed_jobs.extend(failures)
        return rows, failed_jobs

    cpu_workers = max(1, worker_count - 1)
    mps_workers = 1
    cpu_rows, cpu_failures, cpu_completed = _run_job_batch(
        cpu_jobs,
        max_workers=cpu_workers,
        total_jobs=total_jobs,
        starting_completed=completion_count,
        run_state_path=run_state_path,
        state=state,
        fail_fast=fail_fast,
        executor_label="cpu",
    )
    rows.extend(cpu_rows)
    failed_jobs.extend(cpu_failures)
    completion_count = cpu_completed
    if fail_fast and failed_jobs:
        return rows, failed_jobs
    mps_rows, mps_failures, _mps_completed = _run_job_batch(
        mps_jobs,
        max_workers=mps_workers,
        total_jobs=total_jobs,
        starting_completed=completion_count,
        run_state_path=run_state_path,
        state=state,
        fail_fast=fail_fast,
        executor_label="mps",
    )
    rows.extend(mps_rows)
    failed_jobs.extend(mps_failures)
    return rows, failed_jobs


def _run_job_batch(
    jobs: list[BenchmarkJob],
    *,
    max_workers: int,
    total_jobs: int,
    starting_completed: int,
    run_state_path: Path,
    state: dict[str, object],
    fail_fast: bool,
    executor_label: str = "default",
) -> tuple[list[dict[str, object]], list[dict[str, str]], int]:
    rows: list[dict[str, object]] = []
    failed_jobs: list[dict[str, str]] = []
    completed = starting_completed
    if not jobs:
        return rows, failed_jobs, completed
    if max_workers <= 1 or not process_pool_supported():
        return _run_job_batch_serial(
            jobs,
            total_jobs=total_jobs,
            starting_completed=starting_completed,
            run_state_path=run_state_path,
            state=state,
            fail_fast=fail_fast,
            executor_label=f"{executor_label}_serial",
        )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job: dict[Future[dict[str, object]], BenchmarkJob] = {}
        for job in jobs:
            state["jobs"][job.job_id] = {
                "status": "running",
                "output_dir": str(job.run_output_dir),
                "executor": executor_label,
            }
            future_to_job[executor.submit(_execute_benchmark_job, job)] = job
        _write_run_state(run_state_path, state)

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                payload = future.result()
            except Exception as exc:  # pragma: no cover - failure path
                failed_jobs.append({"job_id": job.job_id, "error": str(exc)})
                state["jobs"][job.job_id] = {
                    "status": "failed",
                    "output_dir": str(job.run_output_dir),
                    "error": str(exc),
                }
                state["counts"]["failed"] = int(state["counts"]["failed"]) + 1
                _write_run_state(run_state_path, state)
                print(
                    f"[{completed}/{total_jobs}] failed {job.scenario_id} seed={job.seed} policy={job.policy}: {exc}"
                )
                if fail_fast:
                    for pending_future in future_to_job:
                        pending_future.cancel()
                    break
                continue

            rows.append(dict(payload["row"]))
            completed += 1
            state["jobs"][job.job_id] = {
                "status": "completed",
                "output_dir": str(job.run_output_dir),
                "duration_seconds": payload["duration_seconds"],
            }
            state["counts"]["completed"] = int(state["counts"]["completed"]) + 1
            _write_run_state(run_state_path, state)
            print(
                f"[{completed}/{total_jobs}] completed {job.scenario_id} seed={job.seed} policy={job.policy}"
            )

    return rows, failed_jobs, completed


def _run_job_batch_serial(
    jobs: list[BenchmarkJob],
    *,
    total_jobs: int,
    starting_completed: int,
    run_state_path: Path,
    state: dict[str, object],
    fail_fast: bool,
    executor_label: str,
) -> tuple[list[dict[str, object]], list[dict[str, str]], int]:
    rows: list[dict[str, object]] = []
    failed_jobs: list[dict[str, str]] = []
    completed = starting_completed
    for job in jobs:
        state["jobs"][job.job_id] = {
            "status": "running",
            "output_dir": str(job.run_output_dir),
            "executor": executor_label,
        }
        _write_run_state(run_state_path, state)
        try:
            payload = _execute_benchmark_job(job)
        except Exception as exc:  # pragma: no cover - failure path
            failed_jobs.append({"job_id": job.job_id, "error": str(exc)})
            state["jobs"][job.job_id] = {
                "status": "failed",
                "output_dir": str(job.run_output_dir),
                "error": str(exc),
            }
            state["counts"]["failed"] = int(state["counts"]["failed"]) + 1
            _write_run_state(run_state_path, state)
            print(f"[{completed}/{total_jobs}] failed {job.scenario_id} seed={job.seed} policy={job.policy}: {exc}")
            if fail_fast:
                break
            continue
        rows.append(dict(payload["row"]))
        completed += 1
        state["jobs"][job.job_id] = {
            "status": "completed",
            "output_dir": str(job.run_output_dir),
            "duration_seconds": payload["duration_seconds"],
        }
        state["counts"]["completed"] = int(state["counts"]["completed"]) + 1
        _write_run_state(run_state_path, state)
        print(f"[{completed}/{total_jobs}] completed {job.scenario_id} seed={job.seed} policy={job.policy}")
    return rows, failed_jobs, completed


def _execute_benchmark_job(job: BenchmarkJob) -> dict[str, object]:
    started = time.time()
    experiment_config = load_experiment_config(job.scenario_config)
    seeded_config = _override_seed(experiment_config, job.seed)
    policy_config = _override_policy(seeded_config, job.policy, job.policy_artifact_path)
    result, written_paths = run_experiment_from_config(
        config=policy_config,
        output_dir_override=job.run_output_dir,
        force_write_plots=job.force_write_plots,
        runtime_device=job.runtime_device,
    )
    row = _build_summary_row(
        job,
        metrics=build_simulation_metric_record(result),
        artifact_columns={f"{label}_path": str(path) for label, path in written_paths.items()},
    )
    return {
        "job_id": job.job_id,
        "row": row,
        "duration_seconds": time.time() - started,
    }


def _build_benchmark_jobs(
    benchmark_config: BenchmarkConfig,
    *,
    benchmark_root: Path,
    force_write_plots: bool,
) -> tuple[list[BenchmarkJob], dict[str, list[int]], dict[str, str]]:
    jobs: list[BenchmarkJob] = []
    scenario_seed_map: dict[str, list[int]] = {}
    config_sources: dict[str, str] = {}

    for scenario_path in benchmark_config.scenario_configs:
        experiment_config = load_experiment_config(scenario_path)
        config_sources[str(scenario_path)] = scenario_path.read_text(encoding="utf-8")
        seeds = benchmark_config.seeds or (experiment_config.demand.seed,)
        scenario_seed_map[experiment_config.name] = list(seeds)
        for seed in seeds:
            seeded_config = _override_seed(experiment_config, seed)
            for policy in benchmark_config.policies:
                policy_config = _override_policy(
                    seeded_config,
                    policy,
                    benchmark_config.policy_artifacts.get(policy),
                )
                run_output_dir = benchmark_root / experiment_config.name / f"seed_{seed}" / policy
                jobs.append(
                    BenchmarkJob(
                        benchmark_name=benchmark_config.name,
                        scenario_family=benchmark_config.scenario_family,
                        scenario_id=experiment_config.name,
                        scenario_name=experiment_config.name,
                        scenario_config=scenario_path,
                        seed=seed,
                        policy=policy,
                        policy_family=_policy_family(policy),
                        policy_role=_policy_role(policy),
                        coordination_mode=str(policy_config.simulation.coordination_mode),
                        execution_model=str(policy_config.simulation.execution_model),
                        motion_model=(
                            policy_config.coordination.motion_model
                            if policy_config.coordination is not None
                            else "graph_embedded"
                        ),
                        fleet_size=experiment_config.robots.count,
                        demand_mean_interval=experiment_config.demand.mean_interval,
                        demand_horizon_seconds=experiment_config.demand.horizon_seconds,
                        layout_rows=experiment_config.layout.rows,
                        layout_columns=experiment_config.layout.columns,
                        blocked_cell_count=len(experiment_config.layout.blocked_cells),
                        directed_edge_count=len(experiment_config.layout.directed_edges),
                        topology_difficulty=_topology_difficulty(experiment_config),
                        run_output_dir=run_output_dir,
                        force_write_plots=force_write_plots,
                        policy_artifact_path=benchmark_config.policy_artifacts.get(policy),
                        runtime_device=resolve_runtime_device(
                            policy,
                            use_mps_for_learned_policies=benchmark_config.use_mps_for_learned_policies,
                        ),
                    )
                )
    jobs.sort(key=lambda job: (job.scenario_id, job.seed, job.policy))
    return jobs, scenario_seed_map, config_sources


def _build_summary_row(
    job: BenchmarkJob,
    *,
    metrics: dict[str, object],
    artifact_columns: dict[str, str],
) -> dict[str, object]:
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "benchmark_name": job.benchmark_name,
        "scenario_family": job.scenario_family,
        "scenario_id": job.scenario_id,
        "scenario_name": job.scenario_name,
        "scenario_config": str(job.scenario_config),
        "seed": job.seed,
        "policy": job.policy,
        "policy_family": job.policy_family,
        "policy_role": job.policy_role,
        "coordination_mode": job.coordination_mode,
        "execution_model": job.execution_model,
        "motion_model": job.motion_model,
        "fleet_size": job.fleet_size,
        "demand_mean_interval": job.demand_mean_interval,
        "demand_horizon_seconds": job.demand_horizon_seconds,
        "layout_rows": job.layout_rows,
        "layout_columns": job.layout_columns,
        "blocked_cell_count": job.blocked_cell_count,
        "directed_edge_count": job.directed_edge_count,
        "topology_difficulty": job.topology_difficulty,
        "summary_path": str(job.run_output_dir / _ARTIFACT_FILENAMES["summary"]),
        **artifact_columns,
        **metrics,
    }


def _load_completed_row(job: BenchmarkJob) -> dict[str, object]:
    summary_path = job.run_output_dir / _ARTIFACT_FILENAMES["summary"]
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return _build_summary_row(
        job,
        metrics=_build_metric_record_from_saved_run(job.run_output_dir, dict(payload["metrics"])),
        artifact_columns=_artifact_columns_for_output_dir(job.run_output_dir),
    )


def _artifact_columns_for_output_dir(output_dir: Path) -> dict[str, str]:
    return {
        f"{label}_path": str(output_dir / filename)
        for label, filename in _ARTIFACT_FILENAMES.items()
        if (output_dir / filename).exists()
    }


def _build_metric_record_from_saved_run(
    output_dir: Path,
    summary_metrics: dict[str, object],
) -> dict[str, object]:
    executions = _read_csv_rows(output_dir / _ARTIFACT_FILENAMES["executions"])
    queue_snapshots = _read_csv_rows(output_dir / _ARTIFACT_FILENAMES["queue_snapshots"])
    planner_plans = _read_csv_rows(output_dir / _ARTIFACT_FILENAMES["planner_plans"])
    robot_metrics = _read_csv_rows(output_dir / _ARTIFACT_FILENAMES["robot_metrics"])

    turnaround_times = [
        float(row["turnaround_time"])
        for row in executions
        if row.get("turnaround_time") not in {None, ""}
    ]
    queue_lengths = [
        float(row["ready_tasks"])
        for row in queue_snapshots
        if row.get("ready_tasks") not in {None, ""}
    ]
    robot_busy_time = sum(float(row.get("busy_time", 0.0) or 0.0) for row in robot_metrics)
    robot_idle_time = sum(float(row.get("idle_time", 0.0) or 0.0) for row in robot_metrics)
    planner_statuses = [str(row.get("status", "")) for row in planner_plans]
    replanning_epochs = {
        float(row["plan_time"])
        for row in planner_plans
        if row.get("plan_time") not in {None, ""}
    }

    return {
        "throughput": summary_metrics.get("throughput_per_hour"),
        "mean_task_completion_time": summary_metrics.get("average_turnaround_time"),
        "p95_task_completion_time": _percentile(turnaround_times, 95.0),
        "makespan": summary_metrics.get("makespan"),
        "mean_queue_length": summary_metrics.get("average_queue_length"),
        "p95_queue_length": _percentile(queue_lengths, 95.0),
        "robot_idle_fraction": (
            robot_idle_time / (robot_busy_time + robot_idle_time)
            if robot_busy_time + robot_idle_time > 0
            else 0.0
        ),
        "travel_distance_per_completed_task": summary_metrics.get("average_travel_distance_per_task"),
        "realized_waiting_time": sum(
            float(row.get("travel_to_pickup_wait_time", 0.0) or 0.0)
            + float(row.get("travel_to_dropoff_wait_time", 0.0) or 0.0)
            for row in executions
        ),
        "congestion_event_count": summary_metrics.get("blocked_traversal_events_total"),
        "collision_event_count": summary_metrics.get("safety_violations_total"),
        "deadlock_livelock_incident_count": 0,
        "on_time_completion_rate": summary_metrics.get("on_time_completion_rate"),
        "mean_tardiness": summary_metrics.get("mean_tardiness"),
        "p95_tardiness": summary_metrics.get("p95_tardiness"),
        "overdue_task_count": summary_metrics.get("overdue_task_count"),
        "planning_latency": None,
        "replanning_count": len(replanning_epochs),
        "planner_failure_count": summary_metrics.get("planner_failures_total"),
        "timeout_count": sum(status == "timeout" for status in planner_statuses),
        "path_conflict_count_before_resolution": summary_metrics.get("path_conflicts_before_resolution_total"),
        "sipp_wait_insertion_count": summary_metrics.get("sipp_wait_insertions_total"),
        "planner_wait_time_total": summary_metrics.get("planner_wait_time_total"),
        "mapf_solve_success_rate": (
            sum(status == "planned" for status in planner_statuses) / len(planner_statuses)
            if planner_statuses
            else None
        ),
        "reward_mean": None,
        "reward_std": None,
        "policy_entropy": None,
        "invalid_action_rate": None,
        "masked_action_rejection_rate": None,
        "ppo_kl": None,
        "ppo_clip_fraction": None,
        "value_loss": None,
        "generalization_gap_seen_vs_unseen": None,
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _is_completed_run(output_dir: Path) -> bool:
    summary_path = output_dir / _ARTIFACT_FILENAMES["summary"]
    if not summary_path.exists():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("metrics"), dict)


def _initial_run_state(benchmark_name: str, total_jobs: int) -> dict[str, object]:
    return {
        "benchmark_name": benchmark_name,
        "total_jobs": total_jobs,
        "counts": {
            "queued": 0,
            "completed": 0,
            "skipped": 0,
            "failed": 0,
        },
        "jobs": {},
    }


def _write_run_state(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _effective_parallel_workers(configured_workers: int | None) -> int:
    return configured_workers if configured_workers is not None else default_parallel_worker_count()


def process_pool_supported() -> bool:
    try:
        os.sysconf("SC_SEM_NSEMS_MAX")
    except (AttributeError, PermissionError, OSError, ValueError):
        return False
    return True


def _filter_benchmark_config(
    config: BenchmarkConfig,
    *,
    scenario_filters: tuple[str, ...] | None,
    policy_filters: tuple[str, ...] | None,
    seed_filters: tuple[int, ...] | None,
) -> BenchmarkConfig:
    filtered = config
    if scenario_filters:
        allowed_scenarios = set(scenario_filters)
        filtered_paths = tuple(
            path
            for path in filtered.scenario_configs
            if _scenario_matches_filter(path, allowed_scenarios)
        )
        filtered = replace(filtered, scenario_configs=filtered_paths)
    if policy_filters:
        allowed_policies = set(policy_filters)
        filtered = replace(
            filtered,
            policies=tuple(policy for policy in filtered.policies if policy in allowed_policies),
        )
    if seed_filters:
        allowed_seed_set = set(seed_filters)
        allowed_seeds = (
            tuple(seed for seed in filtered.seeds if seed in allowed_seed_set)
            if filtered.seeds is not None
            else tuple(seed_filters)
        )
        filtered = replace(filtered, seeds=allowed_seeds)
    return filtered


def _scenario_matches_filter(path: Path, allowed_scenarios: set[str]) -> bool:
    if path.stem in allowed_scenarios:
        return True
    try:
        return load_experiment_config(path).name in allowed_scenarios
    except Exception:
        return False


def _override_policy(
    config: ExperimentConfig,
    policy: str,
    artifact_path: Path | None = None,
) -> ExperimentConfig:
    policy_model = config.policy_model
    if artifact_path is not None:
        policy_model = (
            replace(policy_model, artifact_path=artifact_path)
            if policy_model is not None
            else PolicyModelConfig(artifact_path=artifact_path)
        )
    return replace(config, simulation=replace(config.simulation, policy=policy), policy_model=policy_model)


def _override_seed(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    return replace(config, demand=replace(config.demand, seed=seed))


def _resolve_benchmark_paths(config: BenchmarkConfig, config_dir: Path) -> BenchmarkConfig:
    resolved_paths = tuple(
        path if path.is_absolute() else (config_dir / path).resolve()
        for path in config.scenario_configs
    )
    resolved_output = config.output_dir if config.output_dir.is_absolute() else (config_dir / config.output_dir).resolve()
    resolved_manifest = (
        None
        if config.artifact_manifest is None
        else (
            config.artifact_manifest
            if config.artifact_manifest.is_absolute()
            else (config_dir / config.artifact_manifest).resolve()
        )
    )
    alias_artifacts = {} if resolved_manifest is None else _policy_artifacts_from_manifest(resolved_manifest)
    resolved_policy_artifacts = {
        policy_name: artifact_path
        if artifact_path.is_absolute()
        else (config_dir / artifact_path).resolve()
        for policy_name, artifact_path in config.policy_artifacts.items()
    }
    for policy_name, artifact_path in alias_artifacts.items():
        resolved_policy_artifacts.setdefault(policy_name, artifact_path)
    return replace(
        config,
        scenario_configs=resolved_paths,
        output_dir=resolved_output,
        policy_artifacts=resolved_policy_artifacts,
        artifact_manifest=resolved_manifest,
    )


def _policy_family(policy: str) -> str:
    if policy in {
        "fifo",
        "random",
        "nearest_robot_task",
        "nearest_task_for_idle_robot",
        "congestion_aware_nearest_robot_task",
    }:
        return "heuristic_dispatch"
    if policy in {
        "trained_linear_model",
        "trained_mlp_model",
        "trained_graph_dispatch_model",
    }:
        return "learned_dispatch"
    if policy == "random_macro":
        return "random_integrated"
    if policy in {"prioritized_sipp_coordinator", "optimal_mapf_coordinator"}:
        return "planner_integrated"
    if policy == "trained_end_to_end_macro_ppo":
        return "learned_integrated"
    return "custom"


def _policy_role(policy: str) -> str:
    if policy in {"fifo", "random", "nearest_robot_task", "nearest_task_for_idle_robot"}:
        return "dispatch_baseline"
    if policy == "congestion_aware_nearest_robot_task":
        return "dispatch_advanced_baseline"
    if policy in {"trained_linear_model", "trained_mlp_model", "trained_graph_dispatch_model"}:
        return "dispatch_learned"
    if policy == "random_macro":
        return "integrated_baseline"
    if policy in {"prioritized_sipp_coordinator", "optimal_mapf_coordinator"}:
        return "integrated_planner"
    if policy == "trained_end_to_end_macro_ppo":
        return "integrated_learned"
    return "custom"


def _topology_difficulty(config: ExperimentConfig) -> str:
    name = config.name
    if "open" in name or len(config.layout.blocked_cells) == 0:
        return "open"
    if "bottleneck" in name:
        return "bottleneck"
    if "crossing" in name:
        return "crossing"
    if "high_fleet" in name or config.robots.count >= max(config.layout.rows, config.layout.columns):
        return "high_fleet_density"
    if "unseen" in name:
        return "generalization"
    return "structured"


def _benchmark_config_snapshot(config: BenchmarkConfig) -> str:
    lines = [
        "[benchmark]",
        f'name = "{config.name}"',
        f'scenario_family = "{config.scenario_family}"',
        "scenario_configs = [",
        *[f'  "{path}",' for path in config.scenario_configs],
        "]",
        "policies = [",
        *[f'  "{policy}",' for policy in config.policies],
        "]",
        f'output_dir = "{config.output_dir}"',
        f"write_plots = {'true' if config.write_plots else 'false'}",
        f"write_manifest = {'true' if config.write_manifest else 'false'}",
        f"resume = {'true' if config.resume else 'false'}",
        f"fail_fast = {'true' if config.fail_fast else 'false'}",
        f"use_mps_for_learned_policies = {'true' if config.use_mps_for_learned_policies else 'false'}",
    ]
    if config.parallel_workers is None:
        lines.append('parallel_workers = "auto"')
    else:
        lines.append(f"parallel_workers = {config.parallel_workers}")
    if config.artifact_manifest is not None:
        lines.append(f'artifact_manifest = "{config.artifact_manifest}"')
    if config.seeds is not None:
        seeds = ", ".join(str(seed) for seed in config.seeds)
        lines.append(f"seeds = [{seeds}]")
    if config.policy_artifacts:
        lines.append("")
        lines.append("[benchmark.policy_artifacts]")
        for policy_name, artifact_path in sorted(config.policy_artifacts.items()):
            lines.append(f'{policy_name} = "{artifact_path}"')
    return "\n".join(lines) + "\n"


def _policy_artifacts_from_manifest(artifact_manifest: Path) -> dict[str, Path]:
    aliases = load_artifact_aliases(artifact_manifest)
    return {
        policy_name: aliases[policy_name]
        for policy_name in (
            "trained_linear_model",
            "trained_mlp_model",
            "trained_graph_dispatch_model",
            "trained_end_to_end_macro_ppo",
        )
        if policy_name in aliases
    }


def _row_sort_key(row: dict[str, object]) -> tuple[str, int, str]:
    return (
        str(row["scenario_id"]),
        int(row["seed"]),
        str(row["policy"]),
    )


def _percentile(values: list[float | int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = max(0, ceil((percentile / 100.0) * len(ordered)) - 1)
    return ordered[min(position, len(ordered) - 1)]
