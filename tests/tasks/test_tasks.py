"""Tests for task models, queues, and demand adapters."""

from __future__ import annotations

import pytest

from warehouse_sim.demand import DemandGenerationConfig, TaskMetadataConfig, generate_task_demand
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.tasks import DemandTaskAdapterConfig, Task, TaskQueue, tasks_from_demand_records
from warehouse_sim.tasks.models import TaskValidationError


def test_task_validation_rejects_invalid_due_time() -> None:
    with pytest.raises(TaskValidationError):
        Task(
            task_id="task_1",
            release_time=10.0,
            pickup_node="r0_c0",
            dropoff_node="r0_c1",
            due_time=5.0,
        )


def test_task_queue_respects_release_time_fifo_order() -> None:
    queue = TaskQueue()
    queue.add_task(Task(task_id="task_2", release_time=5.0, pickup_node="a", dropoff_node="b"))
    queue.add_task(Task(task_id="task_1", release_time=2.0, pickup_node="a", dropoff_node="b"))
    queue.add_task(Task(task_id="task_3", release_time=5.0, pickup_node="a", dropoff_node="b"))

    assert [task.task_id for task in queue.ready_tasks(4.0)] == ["task_1"]
    assert queue.pop_next_ready(5.0).task_id == "task_1"
    assert queue.pop_next_ready(5.0).task_id == "task_2"
    assert queue.pop_next_ready(5.0).task_id == "task_3"
    assert queue.pop_next_ready(5.0) is None


def test_demand_records_can_be_mapped_into_tasks() -> None:
    graph = build_synthetic_grid_layout(
        SyntheticGridLayoutConfig(
            rows=2,
            columns=2,
            zone_labels={(0, 0): "storage_zone", (1, 1): "dropoff_zone"},
        )
    )
    environment = WarehouseEnvironment(graph=graph)
    demand_result = generate_task_demand(
        config=DemandGenerationConfig(min_tasks=0),
        metadata_config=TaskMetadataConfig(
            task_types=("pick",),
            source_zones=("storage_zone",),
            destination_zones=("dropoff_zone",),
            priorities=(2,),
            service_duration_low=45.0,
            service_duration_high=45.0,
        ),
    )

    tasks = tasks_from_demand_records(
        records=demand_result.records[:3],
        environment=environment,
        config=DemandTaskAdapterConfig(
            default_pickup_zone="storage_zone",
            default_dropoff_zone="dropoff_zone",
        ),
    )

    assert [task.task_id for task in tasks] == ["task_1", "task_2", "task_3"]
    assert all(task.pickup_node == "r0_c0" for task in tasks)
    assert all(task.dropoff_node == "r1_c1" for task in tasks)
    assert all(task.priority == 2 for task in tasks)
    assert all(task.service_time_estimate == 45.0 for task in tasks)
    assert tasks[0].metadata["regime"] == "base"


def test_demand_records_preserve_due_time_when_mapping_into_tasks() -> None:
    graph = build_synthetic_grid_layout(
        SyntheticGridLayoutConfig(
            rows=2,
            columns=2,
            zone_labels={(0, 0): "storage_zone", (1, 1): "dropoff_zone"},
        )
    )
    environment = WarehouseEnvironment(graph=graph)
    demand_result = generate_task_demand(
        config=DemandGenerationConfig(min_tasks=0),
        metadata_config=TaskMetadataConfig(
            task_types=("pick",),
            source_zones=("storage_zone",),
            destination_zones=("dropoff_zone",),
            priorities=(2,),
            service_duration_low=45.0,
            service_duration_high=45.0,
            due_time_slack_low=90.0,
            due_time_slack_high=90.0,
        ),
    )

    tasks = tasks_from_demand_records(
        records=demand_result.records[:1],
        environment=environment,
        config=DemandTaskAdapterConfig(
            default_pickup_zone="storage_zone",
            default_dropoff_zone="dropoff_zone",
        ),
    )

    assert tasks[0].due_time is not None
    assert tasks[0].due_time == pytest.approx(tasks[0].release_time + 90.0)
