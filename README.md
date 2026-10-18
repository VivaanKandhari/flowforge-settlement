# FlowForge: settlement under stress

An exploratory payment-network simulator and graph-policy benchmark prepared for Arbiter Hacks V1, General Track. 
## Reproduce

Python 3.10+ and NumPy are required. Install `requirements.txt`, then run:

```
python -m unittest test_engine.py
python experiment.py
```

The experiment rewrites results.json/results.js. Seeds 0–63 train three models, each for 12 epochs with fixed initialization. Test seeds 1000–1047 are held out. Normal-only and stress models each see 64 episodes; stress modes and initial liquidity are crossed. State counts vary with policy trajectories. Edge-only and graph-stress use the same training states and targets. There is no test-set hyperparameter tuning.

## Method

Graph model: normalized, undirected, amount-weighted queued-payment adjacency with self loops; six bank features; trainable 6×12 projection, tanh; concatenated source/destination embeddings plus five payment features; linear ranking head. NumPy implements the gradient. This is supervised distillation, not reinforcement learning. The teacher scores feasible payments with a two-step greedy lookahead using completion, age and deadline urgency. It sees no future arrivals.

An independent feasibility gate enforces arrived payments, current sender availability and sufficient cash. Integer transfers preserve total cash and cannot create credit. The cycle baseline searches whole-payment cycles of length 2–4 and permits simultaneous net settlement, a stronger capability than the other sequential gross policies. Its advantages must not be presented as a fair ranking-only comparison.

## Evaluation

Eight policies: FIFO among feasible payments (no head-of-line blocking), earliest deadline, smallest amount, cycle netting, two-step lookahead, normal-trained graph, stress-trained graph and edge-only stress ablation. Six scenarios: normal, temporary sender outage, delayed outgoing payments, bursts, unseen ring topology and unseen 12-bank size. Every stream has 100 payments and 60 time ticks. Initial cash is 75 units per bank.

Metrics: completed payments, missed deadlines (unfinished counts as missed), and mean delay with unfinished payments censored at tick 60. Conditional mean delay for completed payments is also exported but not used as the primary delay metric. Summary errors are standard errors; paired differences use an approximate normal 95% interval across 48 seeds. These intervals do not quantify simulator-to-real-world uncertainty.

## Scope and novelty

The proposed contribution is a reproducible stress/ablation comparison, not a proven new algorithm or state-of-the-art result. Results may favor simple baselines. No banking data or user finances are used. Assumptions exclude interbank borrowing, credit, collateral, fees, multi-currency operations and capacity limits. A bank's temporary outage blocks outgoing payments only.

Related work: Bank of Canada, Estimating Policy Functions in Payments Systems Using Reinforcement Learning (2021), https://www.bankofcanada.ca/2021/02/staff-working-paper-2021-7/ ; ECB/Bank of Japan Stella report (2017), https://www.ecb.europa.eu/pub/pdf/other/ecb.stella_project_report_september_2017.pdf . A deeper prior-art review is required before claiming research novelty.


Current finding: graph-minus-edge-only paired completion differences have approximate 95% intervals containing zero in all six scenarios. A graph-layer advantage is not established. Under outages, graph-stress averages 80.02 completions, edge-only 79.92, FIFO 79.06, lookahead 81.52 and cycle netting 84.25. Cycle netting has a different settlement capability. These are synthetic results only.


