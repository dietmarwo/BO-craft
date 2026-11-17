# User manual

## Choose a method

Use the native fcmaes surrogate for its tested small deterministic continuous-box scope and low overhead. Use BO-craft when statistical noise, mixed/conditional parameters, black-box constraints or multiple objectives justify Python GP/TPE overhead. Compare at equal scientific target and total cost; more supported options do not imply a universally better optimizer.

## Make a request

Define named parameters and objective directions in bo.problem/0.1. Supply evaluator id/version, units and input hashes for real models. Each observation has immutable id/trial_id, named physical x, status, raw values and available uncertainty. Completed observations contain every objective/constraint metric. Failed, cancelled, pruned and unknown work do not become numerical GP targets.

For known-noise mode provide variance for every observed metric. An aggregate over independent replications records count and variance_kind=mean. Variance of individual outcomes is not variance of their sample mean. `bo_craft.specs.aggregate` requires distinct replicate seeds and preserves source IDs. A physical model must justify independence; the utility cannot infer it.

Repeated use of the same x and simulation run_seed reproduces a sample path. Use separate run seeds for independent replications. The integrated native evaluate bridge supplies a distinct trial seed and preserves provenance. Its current scope is one unconstrained scalar metric and one thread per evaluation point.

## CLI

```bash
bo-craft validate request.json
bo-craft propose request.json --output proposal.json
bo-craft recommend request.json --kind empirical --output empirical.json
bo-craft recommend request.json --kind posterior --output prediction.json
```

Posterior recommendation currently supports unconstrained continuous scalar BoTorch models, including target-fidelity projection. Its interval is conditional model uncertainty, not an independently validated confidence statement. Constrained/multiobjective recommendations are empirical feasible/Pareto observations.

Check capabilities and typed errors. Unsupported mixed BoTorch spaces, missing hypervolume reference points, unsupported fidelity/pending combinations and unknown fields fail before objective evaluation. Numerical failures remain failures. A proposal command's runtime wall limit bounds even backend work that does not support a Python-level interrupt.

## Adaptive campaigns

Install [experiment-craft](https://github.com/dietmarwo/experiment-craft) and this package into one environment; its [adaptive campaigns](https://github.com/dietmarwo/experiment-craft/blob/main/docs/adaptive-bo.md) name the BO-craft revision they are tested with. Start with examples/adaptive_campaign.py, then edit a copy of its manifest. The manifest separates the proposal Python executable/slot/time reservation, evaluator template, scientific evaluation count, concurrency, job count, declared slot-seconds and absolute campaign deadline.

```bash
experiment-craft adaptive-init campaign.json --workspace /absolute/new/campaign \
  --runner /absolute/optimization-run --node-root /absolute/node --mode foreground
experiment-craft adaptive-continue --workspace /absolute/campaign
experiment-craft adaptive-status --workspace /absolute/campaign
experiment-craft adaptive-stop --workspace /absolute/campaign
experiment-craft pause-dispatch --workspace /absolute/campaign --reason maintenance
experiment-craft backup --workspace /absolute/campaign --destination /absolute/backup.zip
```

Use the CLI help for restore/unpause arguments. Foreground mode is a local test path using the existing runtime; it is not a new detached service. Systemd mode requires a commissioned user manager. Generic command jobs currently use the CLI, not optimization-mcp submission tools.

A campaign can have several evaluation jobs and one numerical proposal job in flight. The runtime remains CPU capacity authority. Accepted candidates and unknown jobs hold their scientific reservations. Failure/cancellation still consumed work; a declared slot-second reservation is not measured CPU usage and is not refunded speculatively.

A campaign deadline includes time between client sessions. While the client is absent, only native per-job limits are enforced. Unknown submissions are observed by the same handle; never delete/reset the workspace to obtain a retry. A changed source package/executable is refused: finish the old environment or make a deliberate new campaign with warm observations and new provenance.

## Constraints, objectives and fidelities

Input constraints are exact linear physical-space filters for supported active numeric parameters. Unknown observed constraints use g(x)<=0. A simulator crash is not evidence of infeasibility. Model-based feasibility is probabilistic, and every application still needs an independent check.

For multiple objectives supply a raw-unit reference point before the comparison. Keep directions, normalization and reference point fixed. `recommend` reports empirical feasible nondominated observations; noisy posterior fronts need fresh-set validation.

Multifidelity KG is an experimental scalar, continuous, synchronous adapter. Declare normalized-fidelity cost intercept/weight and the raw target fidelity. It currently rejects constraints and pending evaluations. Compare against target-fidelity-only optimization at equal total cost. A low-fidelity value is not a target-fidelity answer.

## Optional adapters and reproducibility

Install the pinned Vizier extra only when needed. Configure endpoint/study and a tested algorithm; the local DEFAULT GP smoke is not remote outage acceptance. The BO-MCP import bridge expects pre-generated pending suggestions in an existing campaign and does not synchronize results automatically.

Run `pytest`, `ruff check .`, and `python -m build`. The synthetic benchmark harness uses independent watchdogs, two worker processes, immutable cell identities and retained failures. The evidence report includes all 180 scalar confirmation cells and a 15-cell 20D pilot. These are small synthetic examples; they do not establish performance across real engineering experiments or large conditional spaces.
