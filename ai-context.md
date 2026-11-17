# Agent entry point for BO-craft

Read README.md, docs/capabilities.md and docs/user-manual.md before proposing a run. This package is experimental and not published to a package index; install it from this repository. Inspect `bo-craft capabilities` and the JSON schemas; do not invent options.

Use BoTorch for supported expensive continuous noisy/constraint/multiobjective experiments, Optuna TPE for mixed/conditional domains, and keep the native fcmaes surrogate as the low-overhead deterministic baseline (it is far ahead of BoTorch on deterministic continuous boxes). For integer or categorical parameters with little noise, the experimental GP routes (BoTorch mixed, Optuna `sampler: gp`) measured far better than TPE; under noise with a categorical choice they did not (docs/evidence/plan-2026-11-17). Preserve the scientific target, seed/replicate protocol, units, constraints and resource budget. Compare methods with total elapsed/evaluation cost, retaining failures.

Validate the request, run proposal computation through an admitted trusted command when integrated, execute candidates with the existing evaluator, ingest verified observations once, and retain native evidence unchanged. Generic command submission is CLI-only in the inspected runtime. Do not claim an MCP campaign tool exists.

Recommend observed empirical values, the model-ranked observed point (`--kind model`, a second opinion: not reliably better on fresh seeds) or explicitly labeled posterior predictions. For noise, reserve fresh independent replicates; deterministic equality rechecks do not certify expected performance. Follow experiment-craft unknown-work handling and never restart uncertain trials by guessing. Read the capability matrix for external-service and pruning limits.
