"""Simulation engine and result models for the warehouse simulator."""

from warehouse_sim.simulation.models import (
    CoordinationMode,
    CoordinationRuntimeConfig,
    DispatchArcObservationRecord,
    DispatchNodeObservationRecord,
    DispatchTraceRecord,
    ExecutionModel,
    QueueSnapshot,
    SimulationConfig,
    SimulationResult,
    TaskExecution,
)


def run_simulation(*args, **kwargs):
    from warehouse_sim.simulation.engine import run_simulation as _run_simulation

    return _run_simulation(*args, **kwargs)


def run_experiment_from_config(*args, **kwargs):
    from warehouse_sim.simulation.runner import run_experiment_from_config as _run_experiment_from_config

    return _run_experiment_from_config(*args, **kwargs)


def run_experiment_from_path(*args, **kwargs):
    from warehouse_sim.simulation.runner import run_experiment_from_path as _run_experiment_from_path

    return _run_experiment_from_path(*args, **kwargs)


def run_benchmark_from_config(*args, **kwargs):
    from warehouse_sim.simulation.benchmark import run_benchmark_from_config as _run_benchmark_from_config

    return _run_benchmark_from_config(*args, **kwargs)


def run_benchmark_from_path(*args, **kwargs):
    from warehouse_sim.simulation.benchmark import run_benchmark_from_path as _run_benchmark_from_path

    return _run_benchmark_from_path(*args, **kwargs)


__all__ = [
    "DispatchArcObservationRecord",
    "DispatchNodeObservationRecord",
    "DispatchTraceRecord",
    "CoordinationMode",
    "CoordinationRuntimeConfig",
    "ExecutionModel",
    "QueueSnapshot",
    "SimulationConfig",
    "SimulationResult",
    "TaskExecution",
    "run_benchmark_from_config",
    "run_benchmark_from_path",
    "run_experiment_from_config",
    "run_experiment_from_path",
    "run_simulation",
]
