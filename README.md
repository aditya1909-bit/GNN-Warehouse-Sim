# GNN-Warehouse-Sim

Logic-first warehouse demand modeling scaffold for stochastic warehouse simulation.

## 1) Generate synthetic task demand

```bash
python3 scripts/generate_task_demand.py
```

Default output: `data/task_demand.csv`

Key behavior:
- Base arrivals follow an Exponential interarrival process.
- Morning rush increases arrival rate.
- Lunch break enforces a zero-arrival period.

Useful options:

```bash
python3 scripts/generate_task_demand.py \
  --horizon-seconds 28800 \
  --mean-interval 10 \
  --rush-start 1800 --rush-end 7200 --rush-multiplier 2.0 \
  --lunch-start 14400 --lunch-end 16200 \
  --seed 7
```

## 2) Analyze input distribution

Open `notebooks/input_modeling_analysis.ipynb` and run all cells.

The notebook includes:
- Data card and context for interarrival modeling.
- Histogram and box plot EDA.
- Distribution fitting for Exponential, Gamma, and Weibull.
- Goodness-of-fit metrics: Chi-square, K-S, and Anderson-Darling statistic.
