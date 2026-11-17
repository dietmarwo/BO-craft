# Changelog

## [Unreleased]

Experimental 0.1.0 source; nothing is tagged or published.

### Added

- The [improvement plan of 17 November](docs/plans/bo-craft-improvement-plan-2026-11-17.md), with a [predeclared benchmark protocol and its results](docs/evidence/plan-2026-11-17/README.md).
- Benchmark harness: `benchmarks/problems.py` (eight problems with known optima), `benchmarks/panel.py` (paired arms with a shared initial design and noise stream; latent scoring), `benchmarks/summarize.py` (paired bootstrap ratios, gates, a verifier) and `benchmarks/async_sim.py` (proposal look-ahead with a modelled clock).
- `recommend --kind model`: observed points ranked by a fitted GP's posterior mean, with feasibility probabilities and posterior-mean Pareto sets. On fresh seeds it was not reliably better than the empirical choice (gate G1 failed); it is a documented second opinion.
- Mixed spaces in the BoTorch backend: integers and categoricals through `MixedSingleTaskGP` and BoTorch's alternating mixed search. Optuna's GP sampler is available as `sampler: gp`. Both routes were far better than TPE on three of four mixed strata but not on noisy categorical Branin, so gates G3a/G3b failed and TPE stays the documented default.
- Settings:
  - `acq_maxiter`, separate from `fit_maxiter`;
  - `outcome_transform` (`log`, `bilog`);
  - `replication` (`incumbent`), `replication_threshold`, `max_replicates`;
  - `trust_region`, `num_fantasies` and `feasibility_probability`.
- Proposals may declare `replicate_of`. A repeated point is accepted only as a declared replicate of a completed observation.
- Model diagnostics in every model proposal: lengthscales, noise variance and exact leave-one-out residuals, with warnings.
- Pilots on development seeds (no gates): outcome transforms cut Rosenbrock-8 regret about 20-fold; the model answer halved the noisy Branin–Currin hypervolume regret; the trust region gave no gain on Ackley-20.
- [Pruning and service adapters](docs/pruning-and-services.md): why pruning is not wired and why a campaign-scoped Vizier study is not built, and what each would need.
- Added strict scientific contracts, BoTorch GP/constraint/multiobjective/multifidelity proposals, Optuna mixed/conditional TPE, empirical and scalar posterior recommendations, replication and pruning utilities.
- Added optional Vizier study projections and an explicitly read-only BO-MCP import bridge.
- Added real runtime campaign examples, bounded numerical tests, retained confirmation data and architecture/user documentation.
- experiment-craft's adaptive campaigns (merged there separately) call this package for proposals and keep the trial ledger.

### Changed

- Training selection keeps the best quarter of `training_cap` plus the latest observations, instead of only the latest. Below the cap nothing changes.
- Acquisition defaults stay 4 restarts, 128 raw samples and 50 iterations. Rule D1 of the protocol found no larger setting within 5% of the best in every stratum.
- The BoTorch backend refuses only conditional spaces now, not integer or categorical ones.

### Fixed

- `fit_maxiter` also capped the acquisition optimizer's iterations. The 0.1.0 benchmark's `fit_maxiter: 8` therefore also limited acquisition optimization to 8 iterations.

### Fixed (review of the first source delivery)

- A failed GP hyperparameter fit raises the typed error `model_fit_failed` instead of an untyped BoTorch exception, so the CLI keeps its JSON error contract.
- The CLI reports a missing or unreadable file as `io_error` and a missing optional backend as `missing_dependency`; both were `numerical_failure` or a traceback.
- Vizier suggestions are returned in the problem's declared types: an integral float for an integer parameter becomes an int, and a non-integral one is a `backend_protocol` error instead of an invalid proposal.
- Pruning requires integer steps in completed learning curves too. JSON string keys selected the wrong last step ("9" sorts after "10").
- The delivered LICENSE dropped the MIT liability clause; the repository's complete MIT license is kept.
- CI installs the tested pins from `requirements-tested.txt` and also checks formatting, the recorded evidence, the benchmark smoke profile and a minimal install without Torch.
