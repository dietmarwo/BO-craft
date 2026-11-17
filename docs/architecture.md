# Architecture and contracts

BO-craft owns numerical recommendations and typed scientific data. It has no production execution supervisor or trial database. Its Python API accepts explicit history, pending points, seed and configuration; the integrated campaign ledger belongs to experiment-craft.

## Data flow

A versioned bo.request/0.1 contains bo.problem/0.1, observations, pending candidates and algorithm settings. Strict JSON schemas reject unknown fields. Semantic checks enforce finite values, variance meanings, active parameter sets, metric names, bounds, acyclic conditional dependencies and supported combinations.

The backend returns candidates and diagnostics. The public dispatcher verifies cardinality, physical-space feasibility and duplicates against the full completed/pending archive, then writes a bo.proposal/0.1 artifact bound to request and problem digests. The library does not automatically retry a numerical failure or fabricate a large finite penalty.

BoTorch uses float64 CPU tensors and a declared model space: continuous inputs on [0, 1] (log-scaled when declared), integers as their allowed normalized values, categoricals as indices with a categorical kernel. Objectives may pass through a declared monotone transform (log, bilog) before affine standardization; directions are explicit and metric uncertainty is raw-unit (known variances follow the transform by the delta method). Known variance, inferred scalar noise and deterministic regularization are different choices. Training selection keeps the best quarter of training_cap (default 256) plus the latest completed observations, so the incumbent stays in the model. Historical observations remain in the caller ledger. Covariance cost is still cubic in training size; higher caps require workload measurements.

Optuna reconstructs a TPE study from immutable observations and pending trials. This is a documented algorithm policy: it is not a byte-for-byte resume of an interrupted native Optuna sampler. Integer/category/conditional semantics stay with the backend; inactive parameters are absent. TPE does not infer a GP observation-noise model.

## Persistence and provenance

Proposal artifacts are JSON, written atomically with fsync; no untrusted pickle or executable model state is loaded. Fitted GPs are reconstructed from the authoritative observations. This increases cold-start cost but avoids a second optimizer-state authority. Exact backend-state resume is not advertised.

experiment-craft persists proposal/evaluation intents, stable native request keys, accepted trials, independent replicate seeds, ingestion receipts and budget reservations in its SQLite WAL/FULL decision store. It binds the runner/node, relevant worker identities, Python executable, BO source files and package versions. Input snapshots and output digests are checked. The integrated client never re-submits uncertain work on the assumption that a missing answer proves it never ran.

There is no distributed exactly-once transaction. The guarantees are conservative launch reconciliation and single ingestion of retained outcomes. An intent committed before a crash but never launched remains unknown until runtime/operator evidence resolves it. A launch whose answer was lost can be observed and completed under its existing handle.

Pause, backup and restore reuse the campaign maintenance layer. Restored workspaces remain paused; unpause observes native handles and terminal evidence. Keep only one writable restored campaign copy. Production logout/reboot and remote containment remain runtime commissioning gates.

## Optional services

Vizier keeps one study per campaign: its ID is the configured study name plus a digest of the problem and algorithm, and the request digest is the client_id for outstanding suggestions. Completed observations are imported once with identity/digest metadata; an observation at an outstanding suggestion's point completes that suggestion, and failed or abandoned suggestions are withdrawn. Real local DEFAULT GP and RANDOM_SEARCH service logic was tested. Network-disconnect/remote-service recovery is a separate acceptance gate. NO_ENDPOINT is local testing; production uses a configured persistent service endpoint.

The BO-MCP bridge only queries existing pending suggestions. The inspected generation API has no idempotency key, so automatic generation/result synchronization was not enabled. Its protocol has a contract test; a full live BO-MCP deployment was not commissioned. Treat it as an experimental import bridge, not equivalent to the BoTorch backend.

## Evidence and recommendations

Native results retain their original claims. One seeded evaluation being rechecked does not establish the expectation of a stochastic objective. Empirical recommendations identify historical observations; posterior recommendations identify predictions and conditional GP uncertainty. Fresh independent replicates provide separate validation evidence. No BO result proves global optimality or physical safety.

Multifidelity empirical recommendations use target-fidelity observations only. Posterior recommendations project to target fidelity. Pruning is a single-objective policy: this package decides (`median_decision`) and trains on pruned curves when asked (`pruned: last`); experiment-craft reads the evaluator's progress file through optimization-jobs, records the decision and confirms the cancellation.
