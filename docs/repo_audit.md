# Repository Audit

## Implemented Now

The repository now has eight implemented layers, with the demand/input model still being the most validated component.

- Synthetic task arrivals are generated with a non-homogeneous Poisson process using thinning.
- Base arrivals follow an exponential interarrival process.
- A morning-rush window increases the instantaneous arrival rate.
- A lunch window enforces zero arrivals.
- The default CLI still writes `data/task_demand.csv` with the legacy schema:
  `Task_ID`, `Timestamp`, `Interarrival_Time`, `Regime`.
- Optional task metadata can now be appended to support later task-model work without breaking the default CSV contract.
- Synthetic grid layouts can be built as weighted warehouse graphs.
- Named warehouse zones can be derived from graph nodes or declared explicitly.
- Demand rows can be adapted into explicit task objects and queued by release time.
- Robots, baseline dispatch policies, and a discrete-event simulation engine are implemented.
- Run-level metrics now summarize completion, delay, travel, utilization, and queue behavior.
- TOML experiment configs can drive reproducible runs and write machine-readable outputs.
- Scenario presets and multi-policy benchmark comparisons are now supported.
- Graph and dispatch observations now provide stable typed hooks for future learned policies.
- Observation datasets can now be exported from real dispatch events for future policy-learning work.
- A first observation-driven linear scoring policy can now run inside the simulator from explicit feature weights.
- Plot generation is now part of the standard package dependency path rather than an extra optional environment step.
- The exploratory distribution-fitting workflow remains in the notebook and is intentionally secondary to package code.

## Not Implemented Yet

The repository still does not implement a full warehouse simulator.

- No congestion-aware movement or MAPF
- No battery/charging behavior beyond graph semantics
- No learned policies
- No GNN training, inference, or policy-learning pipeline
- No automatic fitting or training for the linear policy weights

## Current Technical Debt Reduced In Stage 1

- Demand-generation logic is no longer locked inside one script.
- Parameter validation is explicit and rejects overlapping regime windows.
- Reproducibility is first-class through typed configs and seeded generation.
- CSV schema is documented instead of implicit.
- The graph and task layers now have concrete, typed interfaces instead of placeholders.
- The simulation layer now has a real baseline engine instead of package stubs.
- The learned-policy preparation layer now has explicit observation contracts instead of requiring access to engine internals.
- Real dispatch traces can now be exported as candidate-pair datasets instead of being reconstructed later from partial logs.
- The policy layer now supports a real model-configured scorer instead of only hard-coded heuristics.
- Tests now cover demand, graph, environment, task, policy, simulation, and benchmark behavior.

## Migration Notes

- `scripts/generate_task_demand.py` remains available and still works from the repository root.
- The generator logic now lives under `src/warehouse_sim/demand/`.
- The graph layer now lives under `src/warehouse_sim/graph/`.
- The task and queue layer now lives under `src/warehouse_sim/tasks/`.
- The simulation engine now lives under `src/warehouse_sim/simulation/`.
- Config-driven experiment loading now lives under `src/warehouse_sim/config/`.
- Benchmark execution now lives under `src/warehouse_sim/simulation/benchmark.py`.
- Policy observation hooks now live under `src/warehouse_sim/policies/observation.py`.
- Future simulation layers should import from package modules rather than adding business logic to notebooks or scripts.
- Notebook analysis remains valid, but should be treated as downstream analysis of generated data, not as the source of core model logic.
