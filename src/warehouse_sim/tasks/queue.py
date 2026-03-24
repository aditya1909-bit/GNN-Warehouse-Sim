"""Task queue abstractions for staged warehouse simulations."""

from __future__ import annotations

from dataclasses import dataclass

from warehouse_sim.tasks.models import Task


@dataclass(frozen=True)
class _QueuedTask:
    release_time: float
    insertion_order: int
    task: Task


class TaskQueue:
    """FIFO queue with release-time gating for warehouse tasks."""

    def __init__(self, tasks: tuple[Task, ...] | None = None) -> None:
        self._entries: list[_QueuedTask] = []
        self._counter = 0
        if tasks:
            self.extend(tasks)

    def add_task(self, task: Task) -> None:
        """Insert a task while preserving release-time FIFO ordering."""

        self._entries.append(
            _QueuedTask(
                release_time=task.release_time,
                insertion_order=self._counter,
                task=task,
            )
        )
        self._counter += 1
        self._entries.sort(key=lambda entry: (entry.release_time, entry.insertion_order))

    def extend(self, tasks: tuple[Task, ...]) -> None:
        """Insert multiple tasks into the queue."""

        for task in tasks:
            self.add_task(task)

    def ready_tasks(self, current_time: float) -> tuple[Task, ...]:
        """Return the tasks released by the given simulation time."""

        return tuple(entry.task for entry in self._entries if entry.release_time <= current_time)

    def next_release_time(self, current_time: float) -> float | None:
        """Return the next release time strictly after the current time."""

        for entry in self._entries:
            if entry.release_time > current_time:
                return entry.release_time
        return None

    def pop_next_ready(self, current_time: float) -> Task | None:
        """Pop the next FIFO task that has been released."""

        for index, entry in enumerate(self._entries):
            if entry.release_time <= current_time:
                return self._entries.pop(index).task
        return None

    def remove_task(self, task_id: str) -> Task:
        """Remove a specific task by id."""

        for index, entry in enumerate(self._entries):
            if entry.task.task_id == task_id:
                return self._entries.pop(index).task
        raise KeyError(f"Unknown task_id in queue: {task_id}")

    def pending_tasks(self) -> tuple[Task, ...]:
        """Return all pending tasks in queue order."""

        return tuple(entry.task for entry in self._entries)

    def __len__(self) -> int:
        return len(self._entries)
