# Ato: A Thin Operating Layer

**Minimal reproducibility for ML.**
Tracks config structure, code, and runtime so you can explain why runs differ — without a platform.

```bash
pip install ato
```

```python
# Your experiment breaks. Here's how to debug it:
python train.py manual  # See exactly how configs merged
finder.get_trace_statistics('my_project', 'train_step')  # See which code versions ran
finder.find_similar_runs(run_id=123)  # Find experiments with same structure
```

**One question:** "Why did this result change?"
Ato fingerprints config structure, function logic, and runtime outputs to answer it.

---

## What Ato Is

Ato is a thin layer that fingerprints your config structure, function logic, and runtime outputs.

It doesn't replace your stack; it sits beside it to answer one question: **"Why did this result change?"**

Three pieces, zero coupling:

1. **ADict** — Structural hashing for configs (tracks architecture changes, not just values)
2. **Scope** — Priority-based config merging with reasoning and code fingerprinting
3. **SQLTracker** — Local-first experiment tracking in SQLite (zero setup, zero servers)

Each works alone. Together, they explain why experiments diverge.

**Config is not logging — it's reasoning.**
Ato makes config merge order, priority, and causality visible.

---

## Config Superpowers (That Make Reproducibility Real)

These aren't features. They're how Ato is built:

| Capability | What It Does | Why It Matters |
|------------|--------------|----------------|
| **Structural hashing** | Hash based on keys + types, not values | Detect when experiment **architecture** changes, not just hyperparameters |
| **Priority/merge reasoning** | Explicit merge order with `manual` inspection | See **why** a config value won — trace the entire merge path |
| **Namespace isolation** | Each scope owns its keys | Team/module collisions are impossible — no need for `model_lr` vs `data_lr` prefixes |
| **Code fingerprinting** | SHA256 of function bytecode, not git commits | Track **logic changes** automatically — refactoring doesn't create false versions |
| **Runtime fingerprinting** | SHA256 of actual outputs | Detect silent failures when code is unchanged but behavior differs |

**No dashboards. No servers. No ecosystems.**
Just fingerprints and SQLite.

---

## Works With Your Stack (Keep Hydra, MLflow, W&B, ...)

Ato doesn't compete with your config system or tracking platform.
It **observes and fingerprints** what you already use.

**Compose configs however you like:**
- Load Hydra/OmegaConf configs → Ato fingerprints the final merged structure
- Use argparse → Ato observes and integrates seamlessly
- Import OpenMMLab configs → Ato handles `_base_` inheritance automatically
- Mix YAML/JSON/TOML → Ato is format-agnostic

**Track experiments however you like:**
- Log to MLflow/W&B for dashboards → Ato tracks causality in local SQLite
- Use both together → Cloud tracking for metrics, Ato for "why did this change?"
- Or just use Ato → Zero-setup local tracking with full history

**Ato is a complement, not a replacement.**
No migration required. No lock-in. Add it incrementally.

---

## When to Use Ato

Use Ato when:

- **Experiments diverge occasionally** and you need to narrow down the cause
- **Config include/override order** changes results in unexpected ways
- **"I didn't change the code but results differ"** happens repeatedly (dependency/environment/bytecode drift)
- **Multiple people modify configs** and you need to trace who set what and why
- **You're debugging non-determinism** and need runtime fingerprints to catch silent failures

**Ato is for causality, not compliance.**
If you need audit trails or dashboards, keep using your existing tracking platform.

---

## Non-Goals

Ato is **not**:

- A pipeline orchestrator (use Airflow, Prefect, Luigi, ...)
- A hyperparameter scheduler (use Optuna, Ray Tune, ...)
- A model registry (use MLflow Model Registry, ...)
- An experiment dashboard (use MLflow, W&B, TensorBoard, ...)
- A dataset versioner (use DVC, Pachyderm, ...)

**Ato has one job:** Explain why results changed.
Everything else belongs in specialized tools.

---

## Incremental Adoption (No Migration Required)

You don't need to replace anything. Add Ato in steps:

**Step 1: Fingerprint config structure (zero code changes)**
```python
from ato.adict import ADict

config = ADict(lr=0.001, batch_size=32, model='resnet50')
print(config.get_structural_hash())  # Tracks structure, not values
```

**Step 2: Add code fingerprinting to key functions**
```python
from ato.scope import Scope

scope = Scope()

@scope.trace(trace_id='train_step')
@scope
def train_epoch(config):
    # Your training code
    pass
```

**Step 3: Add runtime fingerprinting to outputs**
```python
@scope.runtime_trace(
    trace_id='predictions',
    init_fn=lambda: np.random.seed(42),  # Fix randomness
    inspect_fn=lambda preds: preds[:100]  # Track first 100
)
def evaluate(model, data):
    return model.predict(data)
```

**Step 4: Inspect config merge order**
```bash
python train.py manual  # See exactly how configs merged
```

**Step 5: Track experiments locally**
```python
from ato.db_routers.sql.manager import SQLLogger

logger = SQLLogger(config)
run_id = logger.run(tags=['baseline'])
# Your training loop
logger.log_metric('loss', loss, step=epoch)
logger.finish(status='completed')
```

---

## Quick Start

**Three lines to tracked experiments:**

```python
from ato.scope import Scope

scope = Scope()

@scope.observe(default=True)
def config(config):
    config.lr = 0.001
    config.batch_size = 32
    config.model = 'resnet50'

@scope
def train(config):
    print(f"Training {config.model} with lr={config.lr}")

if __name__ == '__main__':
    train()
```

**Run it:**
```bash
python train.py                          # Uses defaults
python train.py lr=0.01                  # Override from CLI
python train.py manual                   # See config merge order
```

---

## Table of Contents

- [ADict: Structural Hashing](#adict-structural-hashing)
- [Scope: Config Reasoning](#scope-config-reasoning)
  - [Priority-based Merging](#priority-based-merging)
  - [Config Chaining](#config-chaining)
  - [Lazy Evaluation](#lazy-evaluation)
  - [MultiScope: Namespace Isolation](#multiscope-namespace-isolation)
  - [Config Documentation & Debugging](#config-documentation--debugging)
  - [Code Fingerprinting](#code-fingerprinting)
  - [Runtime Fingerprinting](#runtime-fingerprinting)
- [SQL Tracker: Local Experiment Tracking](#sql-tracker-local-experiment-tracking)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Best Practices](#best-practices)
- [FAQ](#faq)
- [Quality Signals](#quality-signals)
- [Contributing](#contributing)

---

## ADict: Structural Hashing

`ADict` tracks when experiment **architecture** changes, not just hyperparameter values.

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Structural Hashing** | Hash based on keys + types → detect architecture changes |
| **Nested Access** | Dot notation: `config.model.lr` instead of `config['model']['lr']` |
| **Format Agnostic** | Load/save JSON, YAML, TOML, XYZ |
| **Safe Updates** | `update_if_absent()` → merge without overwrites |
| **Auto-nested** | `ADict.auto()` → `config.a.b.c = 1` just works |

### Examples

#### Structural Hashing

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

**Why this matters:**
When results differ, you need to know if the experiment **architecture** changed or just the values.

#### Auto-nested Configs

```python
# ❌ Traditional way
config = ADict()
config.model = ADict()
config.model.backbone = ADict()
config.model.backbone.layers = [64, 128, 256]

# ✅ With ADict.auto()
config = ADict.auto()
config.model.backbone.layers = [64, 128, 256]  # Just works!
```

#### Format Agnostic

```python
# Load/save any format
config = ADict.from_file('config.json')
config.dump('config.yaml')

# Safe updates
config.update_if_absent(lr=0.01, scheduler='cosine')  # Only adds scheduler
```

---

## Scope: Config Reasoning

Scope manages configuration through **priority-based merging** with **full reasoning**.

**Config is not logging — it's reasoning.**
Scope makes merge order, priority, and causality visible.

### Priority-based Merging

```
Default Configs (priority=0)
    ↓
Named Configs (priority=0+)
    ↓
CLI Arguments (highest priority)
    ↓
Lazy Configs (computed after CLI)
```

#### Example

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
```

```bash
python train.py                           # lr=0.001, epochs=100
python train.py high_lr                   # lr=0.01, epochs=100
python train.py high_lr long_training     # lr=0.01, epochs=300
```

### Config Chaining

Chain configs with dependencies:

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

### Lazy Evaluation

**Note:** Lazy evaluation requires Python 3.8 or higher.

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

Manage completely separate configuration namespaces with independent priority systems.

**Use case:** Different teams own different scopes without key collisions.

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
    # Both have 'lr' but in separate namespaces!
    print(f"Model LR: {model.lr}, Data LR: {data.lr}")
```

**Key advantage:** `model.lr` and `data.lr` are completely independent. No naming prefixes needed.

**CLI with MultiScope:**

```bash
# Override model scope only
python train.py model.backbone=%resnet101%

# Override both
python train.py model.backbone=%resnet101% data.dataset=%imagenet%
```

### Config Documentation & Debugging

**The `manual` command** visualizes the exact order of configuration application.

```python
@scope.observe(default=True)
def config(config):
    config.lr = 0.001
    config.batch_size = 32
    config.model = 'resnet50'

@scope.manual
def config_docs(config):
    config.lr = 'Learning rate for optimizer'
    config.batch_size = 'Number of samples per batch'
    config.model = 'Model architecture (resnet50, resnet101, etc.)'
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
model: Model architecture (resnet50, resnet101, etc.)
--------------------------------------------------
```

**Why this matters:**
When debugging "why is this config value not what I expect?", you see **exactly** which function set it and in what order.

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

When you run `python train.py manual`, you see:
```
(The Applying Order of Views)
defaults → experiment_config → another_config → (CLI Inputs) → adaptive_lr
```

Now it's **crystal clear** why `lr=0.1` (from `another_config`) and not `0.01`!

### Code Fingerprinting

Track **logic changes** automatically, ignoring cosmetic edits.

#### Static Tracing (`@scope.trace`)

Generates a fingerprint of the function's **logic**, not its name or formatting:

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

All three produce the **same fingerprint** because the underlying logic is identical.

**When fingerprints change:**

```python
@scope.trace(trace_id='train_step')
@scope
def train_v1(config):
    loss = model(data)
    return loss

@scope.trace(trace_id='train_step')
@scope
def train_v2(config):
    loss = model(data) * 2  # ← Logic changed!
    return loss
```

Now fingerprints differ — you've changed the actual computation.

**Example: Catching refactoring bugs**

```python
# Original implementation
@scope.trace(trace_id='forward_pass')
@scope
def forward(model, x):
    out = model(x)
    return out

# Safe refactoring: Added comments, changed variable name, different whitespace
@scope.trace(trace_id='forward_pass')
@scope
def forward(model,x):
    # Forward pass through model
    result=model(x)  # No spaces
    return result
```

These have **the same fingerprint** because the underlying logic is identical — only cosmetic changes.

```python
# Unsafe refactoring: Logic changed
@scope.trace(trace_id='forward_pass')
@scope
def forward(model, x):
    features = model.backbone(x)  # Now calling backbone + head separately!
    logits = model.head(features)
    return logits
```

This has a **different fingerprint** — the logic changed. If you expected them to be equivalent but they have different fingerprints, you've caught a refactoring bug.

### Runtime Fingerprinting

Track what the function **produces**, not what it does.

#### Runtime Tracing (`@scope.runtime_trace`)

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

# With inspect_fn: Track specific parts of output
@scope.runtime_trace(
    trace_id='predictions',
    inspect_fn=lambda preds: preds[:100]  # Only hash first 100 predictions
)
@scope
def evaluate_large_output(model, data):
    return model.predict(data)

# Advanced: Type-only checking (ignore values)
@scope.runtime_trace(
    trace_id='predictions',
    inspect_fn=lambda preds: type(preds).__name__  # Track output type only
)
@scope
def evaluate_structure(model, data):
    return model.predict(data)
```

**Parameters:**
- `init_fn`: Optional function called before execution (e.g., seed fixing, device setup)
- `inspect_fn`: Optional function to extract/filter what to track (e.g., first N items, specific fields, types only)

Even if code hasn't changed, if predictions differ, the runtime fingerprint changes.

#### When to Use Each

**Use `@scope.trace()` when:**
- You want to track code changes automatically
- You're refactoring and want to isolate performance impact
- You need to audit "which code produced this result?"
- You want to ignore cosmetic changes (comments, whitespace, renaming)

**Use `@scope.runtime_trace()` when:**
- You want to detect **silent failures** (code unchanged, output wrong)
- You're debugging non-determinism
- You need to verify model behavior across versions
- You care about what the function produces, not how it's written

**Use both when:**
- Building production ML systems
- Running long-term research experiments
- Multiple people modifying the same codebase

---

## SQL Tracker: Local Experiment Tracking

Lightweight experiment tracking using SQLite.

### Why SQL Tracker?

- **Zero Setup**: Just a SQLite file, no servers
- **Full History**: Track all runs, metrics, and artifacts
- **Smart Search**: Find similar experiments by config structure
- **Code Versioning**: Track code changes via fingerprints
- **Offline-first**: No network required

### Database Schema

```
Project (my_ml_project)
├── Experiment (run_1)
│   ├── config: {...}
│   ├── structural_hash: "abc123..."
│   ├── Metrics: [loss, accuracy, ...]
│   ├── Artifacts: [model.pt, plots/*, ...]
│   └── Fingerprints: [model_forward, train_step, ...]
├── Experiment (run_2)
└── ...
```

### Usage

#### Logging Experiments

```python
from ato.db_routers.sql.manager import SQLLogger
from ato.adict import ADict

# Setup config
config = ADict(
    experiment=ADict(
        project_name='image_classification',
        sql=ADict(db_path='sqlite:///experiments.db')
    ),
    # Your hyperparameters
    lr=0.001,
    batch_size=32,
    model='resnet50'
)

# Create logger
logger = SQLLogger(config)

# Start experiment run
run_id = logger.run(tags=['baseline', 'resnet50', 'cifar10'])

# Training loop
for epoch in range(100):
    # Your training code
    train_loss = train_one_epoch()
    val_acc = validate()

    # Log metrics
    logger.log_metric('train_loss', train_loss, step=epoch)
    logger.log_metric('val_accuracy', val_acc, step=epoch)

# Log artifacts
logger.log_artifact(run_id, 'checkpoints/model_best.pt',
                   data_type='model',
                   metadata={'epoch': best_epoch})

# Finish run
logger.finish(status='completed')
```

#### Querying Experiments

```python
from ato.db_routers.sql.manager import SQLFinder

finder = SQLFinder(config)

# Get all runs in project
runs = finder.get_runs_in_project('image_classification')
for run in runs:
    print(f"Run {run.id}: {run.config.model} - {run.status}")

# Find best performing run
best_run = finder.find_best_run(
    project_name='image_classification',
    metric_key='val_accuracy',
    mode='max'  # or 'min' for loss
)
print(f"Best config: {best_run.config}")

# Find similar experiments (same config structure)
similar = finder.find_similar_runs(run_id=123)
print(f"Found {len(similar)} runs with similar config structure")

# Trace statistics (code fingerprints)
stats = finder.get_trace_statistics('image_classification', trace_id='model_forward')
print(f"Model forward pass has {stats['static_trace_versions']} versions")
```

### Features

| Feature | Description |
|---------|-------------|
| **Structural Hash** | Auto-track config structure changes |
| **Metric Logging** | Time-series metrics with step tracking |
| **Artifact Management** | Track model checkpoints, plots, data files |
| **Fingerprint Tracking** | Version control for code (static & runtime) |
| **Smart Search** | Find similar configs, best runs, statistics |

---

## Hyperparameter Optimization

Built-in **Hyperband** algorithm for efficient hyperparameter search with early stopping.

### How Hyperband Works

Hyperband uses successive halving:
1. Start with many configs, train briefly
2. Keep top performers, discard poor ones
3. Train survivors longer
4. Repeat until one winner remains

### Basic Usage

```python
from ato.adict import ADict
from ato.hyperopt.hyperband import HyperBand
from ato.scope import Scope

scope = Scope()

# Define search space
search_spaces = ADict(
    lr=ADict(
        param_type='FLOAT',
        param_range=(1e-5, 1e-1),
        num_samples=20,
        space_type='LOG'  # Logarithmic spacing
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

# Create Hyperband optimizer
hyperband = HyperBand(
    scope,
    search_spaces,
    halving_rate=0.3,      # Keep top 30% each round
    num_min_samples=3,     # Stop when <= 3 configs remain
    mode='max'             # Maximize metric (use 'min' for loss)
)

@hyperband.main
def train(config):
    # Your training code
    model = create_model(config.model)
    optimizer = Adam(lr=config.lr)

    # Use __num_halved__ for early stopping
    num_epochs = compute_epochs(config.__num_halved__)

    # Train and return metric
    val_acc = train_and_evaluate(model, optimizer, num_epochs)
    return val_acc

if __name__ == '__main__':
    # Run hyperparameter search
    best_result = train()
    print(f"Best config: {best_result.config}")
    print(f"Best metric: {best_result.metric}")
```

### Parameter Types

| Type | Description | Example |
|------|-------------|---------|
| `FLOAT` | Continuous values | Learning rate, dropout |
| `INTEGER` | Discrete integers | Batch size, num layers |
| `CATEGORY` | Categorical choices | Model type, optimizer |

Space types:
- `LOG`: Logarithmic spacing (good for learning rates)
- `LINEAR`: Linear spacing (default)

### Distributed Search

```python
from ato.hyperopt.hyperband import DistributedHyperBand
import torch.distributed as dist

# Initialize distributed training
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

# Create distributed hyperband
hyperband = DistributedHyperBand(
    scope,
    search_spaces,
    halving_rate=0.3,
    num_min_samples=3,
    mode='max',
    rank=rank,
    world_size=world_size,
    backend='pytorch'
)

@hyperband.main
def train(config):
    # Your distributed training code
    model = create_model(config)
    model = DDP(model, device_ids=[rank])
    metric = train_and_evaluate(model)
    return metric

if __name__ == '__main__':
    result = train()
    if rank == 0:
        print(f"Best config: {result.config}")
```

---

## Best Practices

### 1. Project Structure

```
my_project/
├── configs/
│   ├── default.py       # Default config with @scope.observe(default=True)
│   ├── models.py        # Model-specific configs
│   └── datasets.py      # Dataset configs
├── train.py             # Main training script
├── experiments.db       # SQLite experiment tracking
└── experiments/
    ├── run_001/
    │   ├── checkpoints/
    │   └── logs/
    └── run_002/
```

### 2. Config Organization

```python
# configs/default.py
from ato.scope import Scope
from ato.adict import ADict

scope = Scope()

@scope.observe(default=True)
def defaults(config):
    # Data
    config.data = ADict(
        dataset='cifar10',
        batch_size=32,
        num_workers=4
    )

    # Model
    config.model = ADict(
        backbone='resnet50',
        pretrained=True
    )

    # Training
    config.train = ADict(
        lr=0.001,
        epochs=100,
        optimizer='adam'
    )

    # Experiment tracking
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
    # Setup experiment tracking
    logger = SQLLogger(config)
    run_id = logger.run(tags=[config.model.backbone, config.data.dataset])

    try:
        # Training loop
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
- Code fingerprinting: happens once at decoration time
- Runtime fingerprinting: depends on `inspect_fn` complexity
- SQLite logging: milliseconds per metric

### Can I self-host?

Ato runs entirely locally. There's nothing to host.
If you need centralized tracking, use MLflow/W&B alongside Ato.

---

## Quality Signals

**Every release passes 100+ unit tests.**
No unchecked code. No silent failure.

This isn't a feature. It's a commitment.

When you fingerprint experiments, you're trusting the fingerprints are correct.
When you merge configs, you're trusting the merge order is deterministic.
When you trace code, you're trusting the bytecode hashing is stable.

Ato has zero tolerance for regressions.

Tests cover every module — ADict, Scope, MultiScope, SQLTracker, HyperBand — and every edge case we've encountered in production use.

```bash
python -m pytest unit_tests/  # Run locally. Always passes.
```

**If a test fails, the release doesn't ship. Period.**

**Codebase size:** ~10 Python files
Small, readable, auditable. No magic, no metaprogramming.

---

## Requirements

- Python >= 3.7 (Python >= 3.8 required for lazy evaluation features)
- SQLAlchemy (for SQL Tracker)
- PyYAML, toml (for config serialization)

See `pyproject.toml` for full dependencies.

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

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

---

※ I used claude only for generating some missing test codes and README.
