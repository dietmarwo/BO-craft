# Changelog

## [Unreleased]

Experimental 0.1.0 source; nothing is tagged or published.

### Added

- Added strict scientific contracts, BoTorch GP/constraint/multiobjective/multifidelity proposals, Optuna mixed/conditional TPE, empirical and scalar posterior recommendations, replication and pruning utilities.
- Added optional Vizier study projections and an explicitly read-only BO-MCP import bridge.
- Added real runtime campaign examples, bounded numerical tests, retained confirmation data and architecture/user documentation.
- experiment-craft's adaptive campaigns (merged there separately) call this package for proposals and keep the trial ledger.

### Fixed (review of the first source delivery)

- A failed GP hyperparameter fit raises the typed error `model_fit_failed` instead of an untyped BoTorch exception, so the CLI keeps its JSON error contract.
- The CLI reports a missing or unreadable file as `io_error` and a missing optional backend as `missing_dependency`; both were `numerical_failure` or a traceback.
- Vizier suggestions are returned in the problem's declared types: an integral float for an integer parameter becomes an int, and a non-integral one is a `backend_protocol` error instead of an invalid proposal.
- Pruning requires integer steps in completed learning curves too. JSON string keys selected the wrong last step ("9" sorts after "10").
- The delivered LICENSE dropped the MIT liability clause; the repository's complete MIT license is kept.
- CI installs the tested pins from `requirements-tested.txt` and also checks formatting, the recorded evidence, the benchmark smoke profile and a minimal install without Torch.
