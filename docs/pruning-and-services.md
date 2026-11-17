# Pruning and service adapters: state and next steps

Phase 6 of the [improvement plan](plans/bo-craft-improvement-plan-2026-11-17.md). Nothing here changes behaviour. The note records why each item is where it is, and what a change would need.

## Pruning

`bo_craft.pruning.should_prune` is a library decision. It takes completed learning curves and the running trial's intermediate values, and answers prune or continue with Optuna's median or Hyperband rule. Nothing calls it during a campaign.

**Why it isn't wired:**
- An evaluation in experiment-craft is one optimization-jobs job, either a trusted command or an `fcmaes.evaluate/1` point.
- The runtime retains a job's declared outputs when the job ends. It does not stream intermediate metrics from a running job.
- A prune decision therefore has nothing to act on while the evaluation runs.

**What wiring would need:**
1. **optimization-jobs:** a declared intermediate-metric channel for command jobs. For example, an append-only `progress.jsonl` output that the runner retains as it grows, with a size limit, plus a read command (`jobs progress JOB_ID`) that returns the lines as typed JSON. It must not change the job's terminal contract.
2. **experiment-craft:** poll that channel for running evaluation jobs and call `should_prune`. If it says prune, ask the runtime to cancel the job, and record the trial as `pruned` with the partial cost. A pruned trial is not a numerical observation. Only a confirmed stop releases its reservation.
3. **BO-craft:** none beyond today's function. Pruned trials already enter TPE as `PRUNED` and are excluded from GP training.

**Gate when built:** at equal total evaluation time, pruning shortens time-to-target on a learning-curve problem without raising the regret of the final recommendation. Cancellation must be confirmed in the runtime's records.

## Vizier

The adapter creates one study per request (the study ID includes the request digest). It imports every completed observation into that study and asks it for suggestions. This keeps proposals reconstructible from the ledger, but every proposal re-imports the whole history, and each one leaves a study on the server.

**Why it is unchanged:**
- One study per campaign would also have to complete Vizier's own suggested trials with the observations that came back for them. Otherwise they stay active, and Vizier's designer treats them as pending.
- That needs a mapping from suggestion to observation that survives lost answers. Its tests need a live Vizier service; the optional `vizier` extra (Jax and Flax) is not installed in this repository's CI.
- The plan's gate for this change, a local Vizier test over 50 proposals that imports each observation once, could not be run in the session that implemented the plan.

**What a change would need:**
1. Mark each imported trial with the observation ID, as today.
2. Complete a returned suggestion when an observation with the same point and the proposal's `trial_ids` arrives.
3. Stop suggestions that never come back.
4. Run the 50-proposal test against a local Vizier server.

## BO-MCP

The bridge stays read-only and experimental. It reads pre-generated pending suggestions and never generates or completes them, because the inspected API has no idempotency key. Keeping or removing it is the owner's decision. It has no live acceptance test.
