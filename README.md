# Ato: A Scope-Based Config Layer for ML

**Describe experiments as composable config views. Get reproducibility for free.**

Ato lets you build configs from a sequence of "views" (Scopes) with explicit priorities, lazy evaluation, and Pythonic CLI — then fingerprints configs, code, and runtime so you can explain why runs diverge.

```bash
pip install ato
```

```bash
# Run with different config compositions
python train.py                      # Default views
python train.py high_lr              # Apply high_lr view
python train.py high_lr long_run     # Chain multiple views
python train.py manual               # See exactly how configs merged
```

**One core idea:** Config is not configuration files. Config is **reasoning** — a sequence of transformations with priorities, dependencies, and causality.

---

## What Ato Is Really For

Ato is a thin layer for **config reasoning** with reproducibility built in.

**Describe config as views:**
- Small functions or dicts that modify a shared config with explicit priorities
- Chain dependencies: `chain_with=['base', 'gpu']` auto-applies prerequisites
- Lazy evaluation: compute config values **after** CLI args are applied

**Drive everything from a Pythonic CLI:**
- `python train.py baseline long_run augment` applies views in priority order
- Override anything: `python train.py lr=0.01 batch_size=64`
- No YAML files required (but you can load them if you want)

**Inspect "who changed what":**
- `python train.py manual` shows exact view application order
- Built-in documentation system for config keys
- Full causality trace: see which view set which value and why

**Optionally fingerprint code and runtime:**
- Structural hashing: track when experiment **architecture** changes (not just values)
- Code fingerprints: SHA256 of function logic (ignores comments/whitespace)
- Runtime fingerprints: SHA256 of actual outputs (catches silent failures)

**Config is reasoning, not logging.** Ato makes merge order, priority, and causality visible.

---

## Quick Start: Scope-Based Config

```python
from ato.scope import Scope
from ato.adict import ADict

scope = Scope()

@scope.observe(default=True)
def defaults(config):
    config.lr = 1e-3
    config.epochs = 50
    config.model = 'resnet50'

@scope.observe(priority=1)
def high_lr(config):
    config.lr = 3e-3

@scope.observe(priority=2, chain_with='high_lr')
def long_run(config):
    config.epochs = 200

@scope.manual
def docs(manual):
    manual.lr = 'Learning rate for optimizer'
    manual.epochs = 'Number of training epochs'
    manual.model = 'Model architecture'

@scope
def train(config):
    print(f"Training {config.model} for {config.epochs} epochs with lr={config.lr}")
    # Your training code here

if __name__ == '__main__':
    train()
```

**Run it:**
```bash
python train.py                    # lr=1e-3, epochs=50
python train.py high_lr            # lr=3e-3, epochs=50
python train.py long_run           # lr=3e-3, epochs=200 (chain_with auto-applies high_lr)
python train.py lr=0.01            # Override from CLI
python train.py manual             # See view application order + docs
```

**What just happened:**
1. **Views** are functions that modify config
2. **Priorities** control merge order (higher = applied later)
3. **Dependencies** (`chain_with`) auto-apply prerequisite views
4. **CLI** can invoke views or override values directly
5. **Manual** shows exactly how the config was built

This is **config reasoning**, not config loading.

---

## Reproducibility as a Side Effect

Once you have Scope-based config reasoning, reproducibility becomes trivial.

### Structural Hashing: Track Architecture Changes

```python
from ato.adict import ADict

# Same structure, different values
config1 = ADict(lr=0.1, epochs=100, model='resnet50')
config2 = ADict(lr=0.01, epochs=200, model='resnet101')
print(config1.get_structural_hash() == config2.get_structural_hash())  # True

# Different structure (epochs is str!)
config3 = ADict(lr=0.1, epochs='100', model='resnet50')
print(config1.get_structural_hash() == config3.get_structural_hash())  # False
```

**Why this matters:** When results differ, you need to know if the experiment **architecture** changed or just the values.

### Code Fingerprinting: Track Logic Changes

```python
@scope.trace(trace_id='train_step')
@scope
def train_epoch(config):
    loss = compute_loss(model, data)
    return loss
```

Generates SHA256 of function bytecode. Ignores:
- Comments
- Whitespace
- Variable names
- Function names

Detects:
- Logic changes
- Algorithm modifications
- Control flow changes

### Runtime Fingerprinting: Catch Silent Failures

```python
import numpy as np

@scope.runtime_trace(
    trace_id='predictions',
    init_fn=lambda: np.random.seed(42),  # Fix randomness
    inspect_fn=lambda preds: preds[:100]  # Track first 100 only
)
@scope
def evaluate(model, data):
    return model.predict(data)
```

Generates SHA256 of actual outputs. Catches:
- Non-determinism
- Silent failures (code unchanged, output wrong)
- Dependency drift

**The key insight:** Scope already tracks **how** config was built. Adding code/runtime fingerprints just extends that reasoning to code and outputs.

---

## Table of Contents

- [Core Concepts](#core-concepts)
  - [Scope: Config Reasoning](#scope-config-reasoning)
  - [ADict: Structural Awareness](#adict-structural-awareness)
  - [Priority-Based Merging](#priority-based-merging)
  - [Config Chaining](#config-chaining)
  - [Lazy Evaluation](#lazy-evaluation)
  - [MultiScope: Namespace Isolation](#multiscope-namespace-isolation)
- [Fingerprinting & Reproducibility](#fingerprinting--reproducibility)
  - [Static Tracing (Code)](#static-tracing-code)
  - [Runtime Tracing (Outputs)](#runtime-tracing-outputs)
- [SQL Tracker: Local Experiment Tracking](#sql-tracker-local-experiment-tracking)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Works With Your Stack](#works-with-your-stack)
- [When to Use Ato](#when-to-use-ato)
- [Non-Goals](#non-goals)
- [FAQ](#faq)
- [Best Practices](#best-practices)
- [Quality Signals](#quality-signals)

---

## Core Concepts

### Scope: Config Reasoning

Scope manages configuration through **priority-based merging** with **full reasoning**.

**Priority chain:**
```
Default Views (priority=0)
    ↓
Named Views (priority=0+)
    ↓
CLI Arguments (highest priority)
    ↓
Lazy Views (computed after CLI)
```

#### Example: Priority-Based Merging

```python
from ato.scope import Scope

scope = Scope()

@scope.observe(default=True)  # Always applied
def defaults(config):
    config.lr = 0.001
    config.epochs = 100

@scope.observe(priority=1)  # Applied after defaults
def high_lr(config):
    config.lr = 0.01

@scope.observe(priority=2)  # Applied last
def long_training(config):
    config.epochs = 300

@scope
def train(config):
    print(f"LR: {config.lr}, Epochs: {config.epochs}")

if __name__ == '__main__':
    train()
```

```bash
python train.py                           # lr=0.001, epochs=100
python train.py high_lr                   # lr=0.01, epochs=100
python train.py high_lr long_training     # lr=0.01, epochs=300
python train.py lr=0.1                    # lr=0.1, epochs=100 (CLI wins)
```

### ADict: Structural Awareness

`ADict` is not just a dict. It's a **structure-aware config** with:

**Structural hashing:**
```python
config1 = ADict(lr=0.1, epochs=100)
config2 = ADict(lr=0.01, epochs=200)
# Same structure → same hash (tracks architecture, not values)
```

**Nested access:**
```python
config.model.backbone.layers = [64, 128, 256]  # Just works
```

**Access tracking:**
```python
config = ADict(lr=0.1, epochs=100, unused_key=999)
_ = config.lr
minimal = config.get_minimal_config()  # Only {lr: 0.1}
```

**Freeze/defrost:**
```python
config.freeze()    # Read-only mode
config.defrost()   # Editable mode
```

**Format agnostic:**
```python
config = ADict.from_file('config.yaml')
config.dump('config.json')
```

### Priority-Based Merging

Views are applied in **priority order** (lowest first):

```python
@scope.observe(priority=-1)  # Applied first
def base(config):
    config.lr = 0.001

@scope.observe(priority=0)   # Applied second
def mid(config):
    config.lr = 0.01

@scope.observe(priority=1)   # Applied last
def high(config):
    config.lr = 0.1
```

CLI args always have **highest priority**.

### Config Chaining

Chain views with dependencies:

```python
@scope.observe()
def base_setup(config):
    config.project_name = 'my_project'
    config.data_dir = '/data'

@scope.observe(chain_with='base_setup')  # Automatically applies base_setup first
def advanced_training(config):
    config.distributed = True
    config.mixed_precision = True

@scope.observe(chain_with=['base_setup', 'gpu_setup'])  # Multiple dependencies
def multi_node_training(config):
    config.nodes = 4
    config.world_size = 16
```

```bash
# Calling advanced_training automatically applies base_setup first
python train.py advanced_training
# Results in: base_setup → advanced_training
```

**Why this matters:**
- Explicit dependencies (no more forgetting prerequisites)
- Composable configs (build complex from simple)
- Prevents errors (can't use config without dependencies)

### Lazy Evaluation

**Note:** Requires Python 3.8+

Compute configs **after** CLI args are applied:

```python
@scope.observe()
def base_config(config):
    config.model = 'resnet50'
    config.dataset = 'imagenet'

@scope.observe(lazy=True)  # Evaluated AFTER CLI args
def computed_config(config):
    # Adjust based on dataset
    if config.dataset == 'imagenet':
        config.num_classes = 1000
        config.image_size = 224
    elif config.dataset == 'cifar10':
        config.num_classes = 10
        config.image_size = 32
```

```bash
python train.py dataset=%cifar10% computed_config
# Results in: num_classes=10, image_size=32
```

**Python 3.11+ Context Manager:**

```python
@scope.observe()
def my_config(config):
    config.model = 'resnet50'
    config.num_layers = 50

    with Scope.lazy():  # Evaluated after CLI
        if config.model == 'resnet101':
            config.num_layers = 101
```

### MultiScope: Namespace Isolation

Manage completely separate configuration namespaces:

```python
from ato.scope import Scope, MultiScope

model_scope = Scope(name='model')
data_scope = Scope(name='data')
scope = MultiScope(model_scope, data_scope)

@model_scope.observe(default=True)
def model_config(model):
    model.backbone = 'resnet50'
    model.lr = 0.1  # Model-specific learning rate

@data_scope.observe(default=True)
def data_config(data):
    data.dataset = 'cifar10'
    data.lr = 0.001  # Data augmentation learning rate (no conflict!)

@scope
def train(model, data):  # Named parameters match scope names
    print(f"Model LR: {model.lr}, Data LR: {data.lr}")
```

**Key advantage:** `model.lr` and `data.lr` are completely independent. No naming prefixes needed.

**CLI:**
```bash
python train.py model.backbone=%resnet101% data.dataset=%imagenet%
```

### Config Documentation & Debugging

**The `manual` command** shows exact view application order:

```python
@scope.observe(default=True)
def config(config):
    config.lr = 0.001
    config.batch_size = 32

@scope.manual
def docs(manual):
    manual.lr = 'Learning rate for optimizer'
    manual.batch_size = 'Number of samples per batch'
```

```bash
python train.py manual
```

**Output:**
```
--------------------------------------------------
[Scope "config"]
(The Applying Order of Views)
config → (CLI Inputs)

(User Manuals)
lr: Learning rate for optimizer
batch_size: Number of samples per batch
--------------------------------------------------
```

**Complex example:**

```python
@scope.observe(default=True)
def defaults(config):
    config.lr = 0.001

@scope.observe(priority=1)
def experiment_config(config):
    config.lr = 0.01

@scope.observe(priority=2)
def another_config(config):
    config.lr = 0.1

@scope.observe(lazy=True)
def adaptive_lr(config):
    if config.batch_size > 64:
        config.lr = config.lr * 2
```

Run `python train.py manual`:
```
(The Applying Order of Views)
defaults → experiment_config → another_config → (CLI Inputs) → adaptive_lr
```

Now it's **crystal clear** why `lr=0.1` (from `another_config`)!

---

## Fingerprinting & Reproducibility

### Static Tracing (Code)

Track **logic changes** automatically, ignoring cosmetic edits.

```python
# These three functions have IDENTICAL fingerprints
@scope.trace(trace_id='train_step')
@scope
def train_v1(config):
    loss = model(data)
    return loss

@scope.trace(trace_id='train_step')
@scope
def train_v2(config):
    # Added comments
    loss = model(data)  # Compute loss
    return loss

@scope.trace(trace_id='train_step')
@scope
def completely_different_name(config):
    loss=model(data)  # Different whitespace
    return loss
```

All three produce the **same fingerprint** because the logic is identical.

**When fingerprints change:**

```python
@scope.trace(trace_id='train_step')
@scope
def train_v3(config):
    loss = model(data) * 2  # ← Logic changed!
    return loss
```

Now fingerprints differ.

**Example: Catching refactoring bugs**

```python
# Original
@scope.trace(trace_id='forward')
@scope
def forward(model, x):
    out = model(x)
    return out

# Safe refactoring: same fingerprint
@scope.trace(trace_id='forward')
@scope
def forward(model,x):
    result=model(x)  # Same logic, different style
    return result

# Unsafe refactoring: different fingerprint
@scope.trace(trace_id='forward')
@scope
def forward(model, x):
    features = model.backbone(x)  # Logic changed!
    logits = model.head(features)
    return logits
```

### Runtime Tracing (Outputs)

Track what the function **produces**, not what it does.

```python
import numpy as np

# Basic: Track full output
@scope.runtime_trace(trace_id='predictions')
@scope
def evaluate(model, data):
    return model.predict(data)

# With init_fn: Fix randomness for reproducibility
@scope.runtime_trace(
    trace_id='predictions',
    init_fn=lambda: np.random.seed(42)  # Initialize before execution
)
@scope
def evaluate_with_dropout(model, data):
    return model.predict(data)  # Now deterministic

# With inspect_fn: Track specific parts
@scope.runtime_trace(
    trace_id='predictions',
    inspect_fn=lambda preds: preds[:100]  # Only hash first 100
)
@scope
def evaluate_large_output(model, data):
    return model.predict(data)

# Advanced: Type-only checking
@scope.runtime_trace(
    trace_id='predictions',
    inspect_fn=lambda preds: type(preds).__name__  # Track type only
)
@scope
def evaluate_structure(model, data):
    return model.predict(data)
```

**Parameters:**
- `init_fn`: Called before execution (seed fixing, device setup)
- `inspect_fn`: Extract/filter what to track (first N items, specific fields, types only)

**When to use:**
- **Static tracing** (`@scope.trace`): Track code changes, ignore cosmetic edits
- **Runtime tracing** (`@scope.runtime_trace`): Detect silent failures, debug non-determinism
- **Both**: Production ML systems, long-term experiments, multi-person teams

---

## SQL Tracker: Local Experiment Tracking

Lightweight experiment tracking using SQLite.

### Why SQL Tracker?

- **Zero Setup**: Just a SQLite file, no servers
- **Full History**: Track all runs, metrics, artifacts
- **Smart Search**: Find similar experiments by config structure
- **Code Versioning**: Track code changes via fingerprints
- **Offline-first**: No network required

### Usage

#### Logging Experiments

```python
from ato.db_routers.sql.manager import SQLLogger
from ato.adict import ADict

config = ADict(
    experiment=ADict(
        project_name='image_classification',
        sql=ADict(db_path='sqlite:///experiments.db')
    ),
    lr=0.001,
    batch_size=32,
    model='resnet50'
)

logger = SQLLogger(config)
run_id = logger.run(tags=['baseline', 'resnet50'])

for epoch in range(100):
    train_loss = train_one_epoch()
    val_acc = validate()

    logger.log_metric('train_loss', train_loss, step=epoch)
    logger.log_metric('val_accuracy', val_acc, step=epoch)

logger.log_artifact(run_id, 'checkpoints/model_best.pt', data_type='model')
logger.finish(status='completed')
```

#### Querying Experiments

```python
from ato.db_routers.sql.manager import SQLFinder

finder = SQLFinder(config)

# Get all runs
runs = finder.get_runs_in_project('image_classification')

# Find best run
best_run = finder.find_best_run(
    project_name='image_classification',
    metric_key='val_accuracy',
    mode='max'
)

# Find similar experiments (same config structure)
similar = finder.find_similar_runs(run_id=123)

# Trace statistics (code fingerprints)
stats = finder.get_trace_statistics('image_classification', trace_id='model_forward')
```

---

## Hyperparameter Optimization

Built-in **Hyperband** algorithm with successive halving.

### Basic Usage

```python
from ato.adict import ADict
from ato.hyperopt.hyperband import HyperBand
from ato.scope import Scope

scope = Scope()

search_spaces = ADict(
    lr=ADict(
        param_type='FLOAT',
        param_range=(1e-5, 1e-1),
        num_samples=20,
        space_type='LOG'
    ),
    batch_size=ADict(
        param_type='INTEGER',
        param_range=(16, 128),
        num_samples=5,
        space_type='LOG'
    ),
    model=ADict(
        param_type='CATEGORY',
        categories=['resnet50', 'resnet101', 'efficientnet_b0']
    )
)

hyperband = HyperBand(
    scope,
    search_spaces,
    halving_rate=0.3,
    num_min_samples=3,
    mode='max'
)

@hyperband.main
def train(config):
    model = create_model(config.model)
    optimizer = Adam(lr=config.lr)
    val_acc = train_and_evaluate(model, optimizer)
    return val_acc

if __name__ == '__main__':
    best_result = train()
    print(f"Best config: {best_result.config}")
    print(f"Best metric: {best_result.metric}")
```

---

## Works With Your Stack

Ato doesn't compete with your config system or tracking platform.
It **observes and fingerprints** what you already use.

**Compose configs however you like:**
- Load Hydra/OmegaConf → Ato fingerprints the merged structure
- Use argparse → Ato observes and integrates
- Import OpenMMLab configs → Ato handles `_base_` inheritance
- Mix YAML/JSON/TOML → Ato is format-agnostic

**Track experiments however you like:**
- Log to MLflow/W&B → Ato tracks causality in local SQLite
- Use both together → Cloud for metrics, Ato for "why did this change?"
- Or just Ato → Zero-setup local tracking

**Ato is a complement, not a replacement.**
No migration required. No lock-in. Add it incrementally.

---

## When to Use Ato

Use Ato when:

- **Config merge order matters** and you need to trace causality
- **Multiple people modify configs** and you need to see who set what and why
- **CLI-driven workflows** are easier than editing YAML files
- **Experiments diverge occasionally** and you need to narrow down the cause
- **"I didn't change code but results differ"** happens repeatedly

**Ato is for config reasoning, not compliance.**
If you need audit trails or dashboards, keep using your existing tracking platform.

---

## Non-Goals

Ato is **not**:

- A pipeline orchestrator (use Airflow, Prefect, Luigi)
- A hyperparameter scheduler (use Optuna, Ray Tune)
- A model registry (use MLflow Model Registry)
- An experiment dashboard (use MLflow, W&B, TensorBoard)
- A dataset versioner (use DVC, Pachyderm)

**Ato has one job:** Make config reasoning visible and reproducible.
Everything else belongs in specialized tools.

---

## FAQ

### Does Ato replace Hydra?

No. Hydra and Ato have different config management philosophies.

Hydra focuses on hierarchical composition and overrides.
Ato focuses on priority-based reasoning and causality tracking.

Use them together or separately — both work.

### Does Ato conflict with MLflow/W&B?

No. MLflow/W&B provide dashboards and cloud tracking.
Ato provides local causality tracking (config reasoning + code fingerprinting).

Use them together: MLflow/W&B for metrics/dashboards, Ato for "why did this change?"

### Do I need a server?

No. Ato uses local SQLite. Zero setup, zero network calls.

### Can I use Ato with my existing config files?

Yes. Ato is format-agnostic:
- Load YAML/JSON/TOML → Ato fingerprints the result
- Import OpenMMLab configs → Ato handles `_base_` inheritance
- Use argparse → Ato integrates seamlessly

### What if I already have experiment tracking?

Keep it. Ato complements existing tracking:
- Your tracking: metrics, artifacts, dashboards
- Ato: config reasoning, code fingerprinting, causality

No migration required.

### Is Ato production-ready?

Yes. Ato has ~100 unit tests that pass on every release.
Python codebase is ~10 files — small, readable, auditable.

### What's the performance overhead?

Minimal:
- Config fingerprinting: microseconds
- Code fingerprinting: once at decoration time
- Runtime fingerprinting: depends on `inspect_fn` complexity
- SQLite logging: milliseconds per metric

---

## Best Practices

### 1. Project Structure

```
my_project/
├── configs/
│   ├── default.py       # Default views
│   ├── models.py        # Model-specific views
│   └── datasets.py      # Dataset views
├── train.py             # Main training script
├── experiments.db       # SQLite tracking
└── experiments/
    └── run_001/
```

### 2. Config Organization

```python
# configs/default.py
from ato.scope import Scope
from ato.adict import ADict

scope = Scope()

@scope.observe(default=True)
def defaults(config):
    config.data = ADict(dataset='cifar10', batch_size=32)
    config.model = ADict(backbone='resnet50', pretrained=True)
    config.train = ADict(lr=0.001, epochs=100)
    config.experiment = ADict(
        project_name='my_project',
        sql=ADict(db_path='sqlite:///experiments.db')
    )
```

### 3. Combined Workflow

```python
from ato.scope import Scope
from ato.db_routers.sql.manager import SQLLogger
from configs.default import scope

@scope
def train(config):
    logger = SQLLogger(config)
    run_id = logger.run(tags=[config.model.backbone, config.data.dataset])

    try:
        for epoch in range(config.train.epochs):
            loss = train_epoch()
            acc = validate()
            logger.log_metric('loss', loss, epoch)
            logger.log_metric('accuracy', acc, epoch)
        logger.finish(status='completed')
    except Exception as e:
        logger.finish(status='failed')
        raise e

if __name__ == '__main__':
    train()
```

### 4. Reproducibility Checklist

- ✅ Use structural hashing to track config changes
- ✅ Log all hyperparameters to SQLLogger
- ✅ Tag experiments with meaningful labels
- ✅ Track artifacts (checkpoints, plots)
- ✅ Use lazy configs for derived parameters
- ✅ Document configs with `@scope.manual`
- ✅ Add code fingerprinting to key functions
- ✅ Add runtime fingerprinting to critical outputs

---

## Quality Signals

**Every release passes 100+ unit tests.**
No unchecked code. No silent failure.

When you fingerprint experiments, you're trusting the fingerprints are correct.
When you merge configs, you're trusting the merge order is deterministic.
When you trace code, you're trusting the bytecode hashing is stable.

Ato has zero tolerance for regressions.

```bash
python -m pytest unit_tests/  # Run locally. Always passes.
```

**If a test fails, the release doesn't ship. Period.**

**Codebase size:** ~10 Python files
Small, readable, auditable. No magic, no metaprogramming.

---

## Requirements

- Python >= 3.7 (Python >= 3.8 required for lazy evaluation)
- SQLAlchemy (for SQL Tracker)
- PyYAML, toml (for config serialization)

See `pyproject.toml` for full dependencies.

---

## Contributing

Contributions are welcome! Submit issues or pull requests.

### Development Setup

```bash
git clone https://github.com/Dirac-Robot/ato.git
cd ato
pip install -e .
```

### Running Tests

```bash
python -m pytest unit_tests/
```

---

## License

MIT License
