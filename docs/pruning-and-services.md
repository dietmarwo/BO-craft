# Pruning and service adapters

Phase 6 of the [improvement plan](plans/bo-craft-improvement-plan-2026-11-17.md). The first implementation left pruning and a campaign-scoped Vizier study unbuilt; both are built now. This note records how each works, what was measured and what stays open.

## Pruning

Pruning stops evaluations whose learning curve is worse than the completed trials' curves. It is built across three repositories:

1. **optimization-jobs** (after 0.4.1): a command job may declare a progress file, `"progress": {"path", "max_bytes"}`. The evaluator appends JSON lines to it, and `jobs progress JOB_ID` returns the complete lines while the job runs and after it ends. The file is an observation channel: it is not sealed and has no digest, and the job's outputs and terminal record are unchanged.
2. **experiment-craft**: `pruning: {"policy": "median"}` in an adaptive manifest. The campaign declares `progress.jsonl` for each command evaluation and reads `{"step", "value"}` lines. It asks `median_decision` (below) for each running evaluation, records the decision, and only then asks the runtime to cancel the job. A confirmed cancel makes the trial a `pruned` observation with its curve. Its [guide](https://github.com/dietmarwo/experiment-craft/blob/main/docs/adaptive-bo.md#pruning) has the evaluator contract.
3. **BO-craft**:
   - `bo_craft.pruning.median_decision(completed, intermediate, direction=..., startup_trials=5, warmup_steps=1)` is Optuna's median rule in plain Python, without Optuna. A test compares it with Optuna's `MedianPruner` on 150 random cases.
   - Observations may carry `intermediate`, a map from step (as a string) to value. Only completed and pruned single-objective observations may.
   - `config.pruned` (BoTorch only) chooses how the GP treats pruned trials. `ignore` (the default) drops them. `last` trains on each one's last intermediate value, as a completed value. It needs one unconstrained objective without fidelity.
   - Optuna's TPE receives pruned trials as `PRUNED` with their intermediate values. It ranks them below completed trials, by their last step and value.

`should_prune` (median or Hyperband through Optuna) remains as a library function.

**Why `pruned: last` exists.** In the first smoke run, BoTorch ignored pruned trials and kept proposing where it had just pruned: 63 of 76 trials were pruned, and the final regret was 1.75 against 0.043 without pruning. With `last`, a pruned region counts as poor.

**Measured effect: none shown** (gates G7a/G7b of the [protocol](evidence/plan-2026-11-17/README.md#g7-median-pruning-at-equal-total-evaluation-time), committed before the runs). The simulation uses Hartmann-6 learning curves of 10 steps, a budget of 400 time units (40 full evaluations) and 30 fresh seeds:

| Arm | Trials | Pruned | Time to regret 0.1 (median; censored at 400) | Final regret (geometric mean) |
| --- | --- | --- | --- | --- |
| BoTorch, no pruning | 40 | 0 | 345 (10 censored) | 0.041 |
| BoTorch, median pruning, `pruned: last` | 54.0 | 18.6 | 308.5 (12 censored) | 0.031 |
| TPE, no pruning | 40 | 0 | 400 (30 censored) | 0.59 |
| TPE, median pruning | 48.2 | 11.0 | 400 (30 censored) | 0.46 |

The paired ratios for BoTorch with against without pruning:
- time to target: 0.98 [0.91, 1.05], pruning faster in 10 seeds and slower in 12; G7a needed an upper bound below 1;
- final regret: 0.76 [0.43, 1.29]; G7b needed an upper bound of at most 1.05.

Both gates failed. Pruning ran more trials in the same time, but did not reach the target sooner, and the regret estimate is too wide to rule out a loss. It stays opt-in.

**Open:**
- `fleet` (SSH) placements have no progress command.
- Only the median rule is wired.
- Native `fcmaes.evaluate/1` evaluations have no curve and cannot be pruned.

## Vizier

The adapter keeps one study per campaign. Its ID is the configured study name plus a digest of the problem and the algorithm. Each proposal reconciles the request's ledger with the study and then asks for suggestions:

1. **Import once.** Each imported trial carries the observation ID and digest (namespace `bo-craft`). An observation that is already imported is checked against its digest: a differing one is `study_diverged`. It is not imported again.
2. **Complete Vizier's suggestions.** A completed observation at the point of an outstanding (active or requested) suggestion completes that suggestion, so Vizier's own trials do not stay active. Any other completed observation is added as a completed trial.
3. **Withdraw what never comes back.** A failed, pruned, cancelled or unknown observation deletes the suggestion it came from. So does any outstanding suggestion that is neither pending in the request nor held for this request.
4. **Lost answers.** The client ID is the request digest. Vizier returns a client's outstanding trials first, so repeating a request after a lost answer gets the same suggestions.

The proposal's diagnostics report the study, the trial IDs and the reconciliation counts (`imported`, `completed_suggestions`, `withdrawn`, `already_imported`).

**Tests** (`tests/test_vizier.py`, run in their own CI job with the `vizier` extra) use a real local Vizier server with an in-memory database:
- 50 proposals with an observation after each: every observation is imported exactly once, each candidate completes its own suggestion, and only the last, unobserved suggestion is still active;
- failed and abandoned suggestions are withdrawn;
- a changed observation is `study_diverged`;
- integer and categorical parameters keep their types;
- the in-process `NO_ENDPOINT` service reconciles the same way.

**Notes:**
- `DefaultVizierServer` keeps its default database in a file inside the installed package, so studies survive between processes on a machine. Give tests and throwaway runs an in-memory database (`database_url="sqlite:///:memory:"`).
- With `NO_ENDPOINT` each proposal process starts an empty study and imports the whole ledger, which gives the same result.
- Remote-service outages and network faults are not commissioned.

## BO-MCP

The bridge stays read-only and experimental. It reads pre-generated pending suggestions and never generates or completes them, because the inspected API has no idempotency key. Keeping or removing it is the owner's decision. It has no live acceptance test.
