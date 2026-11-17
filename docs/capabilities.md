# Capability and acceptance matrix

| Feature | State | Evidence / boundary |
|---|---|---|
| Strict JSON/problem/observation contracts | Implemented | Finite/bounds/variance/identity rejection tests |
| Noisy scalar BoTorch | Implemented, experimental integration | Known/inferred modes; 30-seed synthetic panels |
| Pending-aware BoTorch | Implemented | Numerical pending test and real overlapping campaigns |
| Mixed/conditional Optuna TPE | Implemented | Conditional test; scalar TPE confirmation; no GP uncertainty claim |
| Mixed-space GP (BoTorch integers and categoricals) | Implemented, experimental | Gate G3a failed only on noisy categorical Branin; 5–21× better than TPE in the other three strata ([benchmarks of 17 November](evidence/plan-2026-11-17/README.md)) |
| Optuna GP sampler (`sampler: gp`) | Implemented, experimental | Gate G3b failed only on noisy categorical Branin; 2.4–73× better than TPE in the other three strata, about 4× cheaper than BoTorch mixed |
| Model recommendation (`--kind model`) | Implemented | Gate G1 failed: not reliably better than the empirical choice on three noisy strata; a second opinion, not the default answer |
| Fit diagnostics (lengthscales, noise, leave-one-out) | Implemented | Leave-one-out matches refits with fixed hyperparameters (test) |
| Training selection (best quarter plus latest) | Implemented | Incumbent kept beyond `training_cap` (test) |
| Outcome transforms (`log`, `bilog`) | Experimental | Back-transform and positivity tests; pilot on Rosenbrock-8 (10 development seeds): log 0.046 [0.026, 0.083] and bilog 0.052 [0.023, 0.12] times the regret of standardization alone, better in 10/10 seeds; no confirmation |
| Declared replicates (`replication: incumbent`) | Experimental, opt-in | Mixed development results; gate G4 failed (D2 chose a threshold that cannot trigger; such thresholds are now refused) |
| Trust region (`trust_region`) | Experimental | Ledger replay test; pilot on Ackley-20 with 80 calls (10 development seeds): 1.04 [0.93, 1.16] × the regret of plain BoTorch, no gain |
| Known linear input constraints | Implemented for supported active numeric spaces | Candidate and rejection tests |
| Black-box constraints | Implemented BoTorch/TPE | Constrained confirmation; no certified feasibility claim |
| Noisy multiobjective qLogNEHVI | Implemented, bounded numerical smoke | Direction/reference/constraint test; pilot on noisy Branin–Currin (10 development seeds): the model answer halves the hypervolume regret of the empirical front (0.50 [0.35, 0.69]); no confirmation |
| Multifidelity KG / cost | Experimental scalar synchronous adapter | Real numerical test, target-fidelity recommendation semantics; `num_fantasies` configurable; never compared with target-only BO at equal cost |
| Posterior recommendation | Experimental scalar unconstrained BoTorch | Independent linear-solve and prediction tests |
| Replicate aggregation | Implemented utility | Independent-seed and variance-of-mean tests |
| Pruning | Library-only single-objective policy | Median/Hyperband utility; live streaming needs an optimization-jobs channel ([note](pruning-and-services.md)) |
| Vizier | Optional request-scoped adapter | Real local DEFAULT GP and RANDOM_SEARCH; remote faults uncommissioned; a campaign-scoped study is designed but not built ([note](pruning-and-services.md)) |
| BO-MCP | Experimental read-only import | Contract-tested query; no live server acceptance or automatic mutation |
| Higher dimensions | Pilot evidence only | 20D Ackley pilots (0.1.0: 5 seeds, 12 calls; 17 November: 10 seeds, 80 calls, BoTorch 0.56 × random and 0.75 × TPE regret, trust region no gain); no automatic selector/SAAS claim |
| Adaptive campaign durability | Implemented local experiment-craft extension | Lost launch answer, six unique outcomes, no repeated dispatch, paused backup/restore |
| GPU / multi-host commissioning | Outside local acceptance | January hardware gates remain open |
| Hosted/MCP BO service | Not introduced | Existing CLI/runtime contracts reused |

Unsupported combinations are rejected rather than approximated silently. A historical method name in a plan is not an available option. Only documented CLI/schema fields are accepted.
