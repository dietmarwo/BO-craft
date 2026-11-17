# BO-craft

Typed Bayesian optimization proposals for expensive experiments, integrated with experiment-craft and optimization-jobs. This is an experimental source release, not a published package or hosted service.

BoTorch supplies noisy continuous and mixed (integer, categorical) models, constraints, several objectives and bounded scalar multifidelity. Optuna supplies TPE for mixed and conditional spaces, and a GP sampler for unconditional ones. Recommendations are empirical (best observed), model-based (the observed point the fitted GP ranks best) or posterior predictions. A tested optional Vizier adapter uses request-scoped study projections. An experimental read-only BO-MCP bridge can import pre-generated suggestions; it does not generate or synchronize campaigns automatically.

| Layer | Owner |
|---|---|
| Numerical model, acquisition, recommendation, observation contracts | BO-craft |
| Trial IDs, accepted proposals, scientific budgets, continuation | experiment-craft |
| Trusted execution, CPU admission, cancellation and artifacts | optimization-jobs |
| Physical model, units, metric and fidelity meanings | Application repository |

The existing native fcmaes-surrogate remains the lightweight deterministic baseline, and the better choice for deterministic continuous boxes: in fcmaes-rust's comparison it reaches far lower regret at a fraction of the proposal time. BO-craft does not make it a Python dependency or change its job profile. The [user manual](docs/user-manual.md#choose-a-method) has a chooser with the measured evidence.

## Install from this source

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
python -m pip install -e '.[botorch,optuna,test]'
bo-craft capabilities
```

A minimal `pip install .` installs the JSON contracts and CLI without importing Torch. Choose optional backends explicitly. For a reproducible numerical environment use requirements-tested.txt; Vizier's tested compatibility pins are separate because the current upstream unconstrained dependency resolution was incompatible.

See [user manual](docs/user-manual.md), [architecture](docs/architecture.md), [capabilities and limits](docs/capabilities.md), [measured evidence](docs/evidence/README.md) and the [benchmarks of 17 November](docs/evidence/plan-2026-11-17/README.md), and the [agent guide](ai-context.md). Plans are in docs/plans. CI covers CPU numerical tests and package build. No GPU or production fleet claim is made.

## Local library use

```python
from bo_craft import propose
request = {"schema_version":"bo.request/0.1", "seed":41, "q":1,
    "problem":{"schema_version":"bo.problem/0.1",
        "parameters":[{"name":"x","type":"continuous","lower":-2.0,"upper":2.0}],
        "objectives":[{"name":"score","direction":"minimize"}]},
    "config":{"backend":"botorch","noise":"inferred"},
    "observations":[], "pending":[]}
result = propose(request)
print(result["candidates"])
```

The caller executes candidates and supplies typed observations. Results include request/problem digests, exact history/pending IDs, resolved diagnostics and package versions. No arbitrary code is carried in the request.

For a real admitted campaign, install [experiment-craft](https://github.com/dietmarwo/experiment-craft), whose [adaptive campaigns](https://github.com/dietmarwo/experiment-craft/blob/main/docs/adaptive-bo.md) call this package, into the same environment and run:

```bash
python examples/adaptive_campaign.py --runner /absolute/optimization-run \
  --output /absolute/new/noisy-demo
```

The example uses known synthetic noise and independent trial seeds. It checks eight unique observations and no duplicate execution on repeated continuation. Read the manual before adapting it to a physical model.
