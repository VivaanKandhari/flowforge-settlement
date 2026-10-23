# FlowForge — settlement under stress

Target: Arbiter Hacks V1, General Track. Solo entrant: Vivaan Kandhari.

Status: registered, source and demo published; not yet submitted. The submission window opened on 21 October 2026 at 00:00 HKT. This prototype was prepared on 14 October, before that opening; pre-event development is disclosed and no claim of development within the event window is made.

Repository: https://github.com/VivaanKandhari/flowforge-settlement
Demo: https://youtu.be/1fg3Rq3iSpM

## Elevator pitch
An auditable payment-network simulator testing learned graph scheduling, liquidity-saving cycles and simple policies under outages and unseen topologies.

## Inspiration
Payment networks can become gridlocked when cash is trapped among participants. We wanted an experiment that tests a technical hypothesis rather than adding an AI interface to an existing workflow: does graph-based scheduling help when a participant stops paying?

## What it does
FlowForge simulates synthetic interbank payments with integer cash balances, arrival times, deadlines and temporary outgoing-payment outages. Its dashboard replays measured traces, compares eight policies on identical streams, shows bank balances and queues, and exports the complete experiment.

## How it was built
Python and NumPy implement the simulator, graph convolution, supervised distillation and evaluation. A two-step constrained lookahead teacher supplies soft targets. Three models compare normal training, stress training and an edge-only ablation. An independent feasibility gate enforces arrivals, sender availability, nonnegative balances and cash conservation. A cycle-netting baseline permits feasible simultaneous whole-payment cycles of length 2–4. The dashboard uses HTML, CSS, JavaScript and SVG.

## Experiment and results
Training uses seeds 0–63. Evaluation uses 48 held-out seeds per scenario, with six scenarios and eight policies: 2,304 runs. In the outage case, graph-stress averages 80.02 settled payments per 100, compared with 79.92 for the edge-only ablation, 79.06 for FIFO, 81.52 for lookahead and 84.25 for cycle netting. The graph-minus-edge paired difference is 0.10 with an approximate 95% interval from -0.50 to 0.71. Graph-layer advantage is not established in any of the six scenarios. Cycle netting's gain includes a different settlement capability, so it is not a ranking-only comparison.

## Challenges and learning
Avoiding future-arrival leakage, keeping settlement invariants independent of the model, evaluating unfinished payments without survivor bias, and separating algorithm effects from settlement-mechanism effects. Five invariant/reproducibility tests pass. A negative ablation result changed the pitch: this is a transparent research benchmark, not a claimed superior scheduler.

## Proposed contribution
A reproducible comparison of graph messages, stress training and settlement mechanisms under controlled failures. Graph scheduling and liquidity-saving mechanisms already have prior art; we do not claim a new state-of-the-art algorithm or established research novelty.

## Next steps
Investigate learned ranking of feasible netting batches, capacity-constrained settlement, multiple outage patterns, and sensitivity to simulator distributions. Validate hypotheses on an independent development split before evaluating on a new untouched test split.

## AI assistance and limitations
AI tools generated the implementation, ran training/evaluation and created materials. The video uses real prototype screenshots and synthesized narration. The entrant must review and understand the work. All data is synthetic. There are no real bank connections, clinical or financial claims, deployment claims, participant-research claims or guaranteed performance gains.

## Prior art
Bank of Canada (2021), Estimating Policy Functions in Payments Systems Using Reinforcement Learning: https://www.bankofcanada.ca/2021/02/staff-working-paper-2021-7/

ECB / Bank of Japan (2017), Stella project report: https://www.ecb.europa.eu/pub/pdf/other/ecb.stella_project_report_september_2017.pdf
