# Simulation Baseline

## Current Scope

The simulator is still a discrete-event dispatch baseline, but it now has two execution-fidelity layers:

- `idealized`: original independent shortest-path travel calculations
- `reserved_edges` / `reserved_nodes`: realized route execution with simplified shared-resource contention

## Process Overview

At each event time:

1. Ready tasks and idle robots are collected.
2. The dispatch policy selects one robot-task pairing at a time.
3. The engine materializes a shortest path from robot to pickup and pickup to dropoff.
4. The execution model converts those paths into realized timing.
5. The robot remains unavailable until realized completion.
6. The simulation advances to the next task release or robot availability event.

## What Changes In Stage 9

- Travel is no longer only a scalar shortest-path cost.
- Completed executions now carry explicit route node/arc sequences.
- Congestion-aware modes can add waiting when a reserved edge or node is unavailable.
- Those waits affect robot availability, queueing, turnaround time, and benchmark outcomes.

## What The Reservation Models Are

- `reserved_edges`: only one robot may traverse a directed edge segment at a time.
- `reserved_nodes`: nodes behave as single-robot resources, including station-like occupancy during service.

These are simplified reservation models. They do not replan globally and should not be described as MAPF.

## Metrics Affected

The simulator still reports completion, waiting, turnaround, queue length, throughput, and utilization. It now also reports:

- realized travel time total
- realized travel distance total
- congestion delay total
- average congestion delay per completed task
- blocked traversal events total

## Current Simplifications

- no full MAPF or optimal conflict resolution
- no congestion-aware rerouting after assignment
- no battery constraints
- no robot failures or preemption
- no learned dispatch policy
