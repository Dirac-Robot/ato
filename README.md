# Ato: A Tiny Orchestrator

**When your model fails, you need to know why. When configs collide, you need to see where.**

Ato is what happens when experiment tracking stops pretending to be MLOps.

No dashboards. No platforms. No magic.
Just fingerprints of what actually ran — configs, code, outputs.

```bash
pip install ato
```

```python
# Your experiment breaks. Here's how to debug it:
python train.py manual  # See exactly how configs merged
finder.get_trace_statistics('my_project', 'train_step')  # See which code versions ran
finder.find_similar_runs(run_id=123)  # Find experiments with same structure
```

**One-line pitch:** ML experiments fail for three reasons — config changes, code changes, or runtime behavior changes. Ato tracks all three automatically.

---

## What Ato Is

Ato is an orchestration layer for Python experiments.

Three pieces, zero coupling:

1. **ADict** — Config management with structural hashing (track when experiment architecture changes, not just values)
2. **Scope** — Function decoration with priority-based merging, dependency chaining, and automatic code fingerprinting
3. **SQLTracker** — Local-first experiment tracking in SQLite (zero setup, zero servers)

Each works alone. Together, they form a reproducibility engine.

Not for compliance. Not for dashboards. For **debugging experiments when results diverge**.

---

## Why Ato Exists

Most config systems solve merging.
Most tracking systems solve logging.

Ato solves **"why did this experiment produce different results?"**

The answer requires three fingerprints:
- Config structure (did hyperparameters change?)
- Code bytecode (did implementation change?)
- Runtime output (did behavior change?)

Ato tracks all three. Automatically. With zero configuration.

**This isn't a feature. It's an architecture decision.**

**What you get:**
- Configs merge with explicit priority. Conflicts are visible, not silent.
- Code changes are fingerprinted automatically. No git commits required.
- Experiments are tracked in SQLite. No servers, no auth, no network calls.
- Namespace collisions are impossible. Each scope owns its keys.

**What you don't get:**
- Dashboards
- Model registries
- Dataset versioning
- Plugin ecosystems

Ato is a **layer**, not a platform. It works between your tools, not instead of them.

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

**That's it.** No YAML files. No launchers. No setup.

---

## Table of Contents

- [ADict: Enhanced Dictionary](#adict-enhanced-dictionary)
- [Scope: Configuration Management](#scope-configuration-management)
  - [Config Chaining](#config-chaining)
  - [MultiScope: Namespace Isolation](#multiscope-namespace-isolation)
  - [Config Documentation & Debugging](#configuration-documentation--debugging)
  - [Reproducibility Engine](#reproducibility-engine)
- [SQL Tracker: Experiment Tracking](#sql-tracker-experiment-tracking)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [Composability](#composability)

---

## ADict: Enhanced Dictionary

`ADict` is an enhanced dictionary for managing experiment configurations.

### Core Features

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| **Structural Hashing** | Hash based on keys + types, not values | Track when experiment **structure** changes (not just hyperparameters) |
| **Nested Access** | Dot notation for nested configs | `config.model.lr` instead of `config['model']['lr']` |
| **Format Agnostic** | Load/save JSON, YAML, TOML, XYZ | Work with any config format |
| **Safe Updates** | `update_if_absent()` method | Merge configs without accidental overwrites |
| **Auto-nested** | `ADict.auto()` for lazy creation | `config.a.b.c = 1` just works - no KeyError |

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
config.data.augmentation.brightness = 0.2
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

## Scope: Configuration Management

Scope manages configuration through **priority-based merging** and **CLI integration**.

### Key Concept: Priority Chain

```
Default Configs (priority=0)
    ↓
Named Configs (priority=0+)
    ↓
CLI Arguments (highest priority)
    ↓
Lazy Configs (computed after CLI)
```

### Basic Usage

#### Simple Configuration

```python
from ato.scope import Scope

scope = Scope()

@scope.observe()
def my_config(config):
    config.dataset = 'cifar10'
    config.lr = 0.001
    config.batch_size = 32

@scope
def train(config):
    print(f"Training on {config.dataset}")
    # Your code here

if __name__ == '__main__':
    train()
```

#### Priority-based Merging

```python
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

#### CLI Configuration

Override any parameter from command line:

```bash
# Simple values
python train.py lr=0.01 batch_size=64

# Nested configs
python train.py model.backbone=%resnet101% model.depth=101

# Lists and complex types
python train.py layers=[64,128,256,512] dropout=0.5

# Combine with named configs
python train.py my_config lr=0.001 batch_size=128
```

**Note**: Wrap strings with `%` (e.g., `%resnet101%`) instead of quotes.

### Config Chaining

Sometimes configs have dependencies on other configs. Use `chain_with` to automatically apply prerequisite configs:

```python
@scope.observe()
def base_setup(config):
    config.project_name = 'my_project'
    config.data_dir = '/data'

@scope.observe()
def gpu_setup(config):
    config.device = 'cuda'
    config.num_gpus = 4

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

# Calling multi_node_training applies all dependencies
python train.py multi_node_training
# Results in: base_setup → gpu_setup → multi_node_training
```

**Why this matters:**
- **Explicit dependencies**: No more remembering to call prerequisite configs
- **Composable configs**: Build complex configs from simpler building blocks
- **Prevents errors**: Can't use a config without its dependencies

### Lazy Evaluation

**Note:** Lazy evaluation features require Python 3.8 or higher.

Sometimes you need configs that depend on other values set via CLI:

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

**Python 3.11+ Context Manager**:

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

**Use case**: Different teams own different scopes without key collisions.

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

**Key advantage**: `model.lr` and `data.lr` are completely independent. No need for naming conventions like `model_lr` vs `data_lr`.

**CLI with MultiScope:**

```bash
# Override model scope only
python train.py model.backbone=%resnet101%

# Override data scope only
python train.py data.dataset=%imagenet%

# Override both
python train.py model.backbone=%resnet101% data.dataset=%imagenet%
```

### Configuration Documentation & Debugging

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
When debugging "why is this config value not what I expect?", you can see **exactly** which function set it and in what order.

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

### Config Import/Export

```python
@scope.observe()
def load_external(config):
    # Load from any format
    config.load('experiments/baseline.json')
    config.load('models/resnet.yaml')

    # Export to any format
    config.dump('output/final_config.toml')
```

**OpenMMLab compatibility:**

```python
# Import OpenMMLab configs - handles _base_ inheritance automatically
config.load_mm_config('mmdet_configs/faster_rcnn.py')
```

**Hierarchical composition:**

```python
from ato.adict import ADict

# Load configs from directory structure
config = ADict.compose_hierarchy(
    root='configs',
    config_filename='config',
    select={
        'model': 'resnet50',
        'data': 'imagenet'
    },
    overrides={
        'model.lr': 0.01,
        'data.batch_size': 64
    },
    required=['model.backbone', 'data.dataset'],  # Validation
    on_missing='warn'  # or 'error'
)
```

### Argparse Integration

```python
from ato.scope import Scope
import argparse

scope = Scope(use_external_parser=True)
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)

@scope.observe(default=True)
def config(config):
    config.lr = 0.001
    config.batch_size = 32

@scope
def train(config):
    print(f"GPU: {config.gpu}, LR: {config.lr}")

if __name__ == '__main__':
    parser.parse_args()  # Merges argparse with scope
    train()
```

### Reproducibility Engine

**The Question:** "Why did my experiment produce different results?"

When results diverge, you need to know:
- Did the code change?
- Did the config structure change?
- Did the runtime behavior change?

**Ato tracks three dimensions of reproducibility:**

| Dimension | What Changes | How Ato Tracks It |
|-----------|--------------|-------------------|
| **Config** | Hyperparameters, model architecture | Structural hashing (ADict) |
| **Code** | Function implementation, logic | Static tracing (`@scope.trace`) |
| **Output** | Model predictions, training dynamics | Runtime tracing (`@scope.runtime_trace`) |

This isn't just versioning — it's a **causal debugging system** for experiments.

#### Example: Full Reproducibility Tracking

```python
from ato.scope import Scope
from ato.db_routers.sql.manager import SQLLogger, SQLFinder

scope = Scope()

@scope.observe(default=True)
def config(config):
    config.model = 'resnet50'
    config.lr = 0.001
    config.batch_size = 32
    config.experiment = {'project_name': 'my_project', 'sql': {'db_path': 'sqlite:///exp.db'}}

# Track code changes
@scope.trace(trace_id='train_step')
def train_epoch(model, data):
    # Training logic here
    return loss

# Track output changes
@scope.runtime_trace(
    trace_id='model_predictions',
    inspect_fn=lambda preds: preds[:100]  # Track first 100 predictions
)
def evaluate(model, test_data):
    predictions = model.predict(test_data)
    return predictions

@scope
def train(config):
    logger = SQLLogger(config)
    run_id = logger.run(tags=['baseline', 'resnet50'])

    model = create_model(config.model)

    for epoch in range(100):
        loss = train_epoch(model, train_data)
        logger.log_metric('loss', loss, step=epoch)

    preds = evaluate(model, test_data)

    logger.finish(status='completed')

if __name__ == '__main__':
    train()
```

**What you get:**

1. **Config fingerprint** (structural hash):
   - Tracks when experiment **architecture** changes
   - Not just values — detects when you add/remove keys or change types

2. **Code fingerprint** (static trace):
   - SHA256 hash of function bytecode, constants, variables
   - Changes when you modify `train_epoch()` logic
   - Query: "Show me all experiments that used v1 vs v2 of train_step"

3. **Output fingerprint** (runtime trace):
   - SHA256 hash of actual predictions/outputs
   - Detects silent failures (code unchanged, output different)
   - Query: "Why do my predictions differ when config/code are identical?"

#### Debugging with Reproducibility Data

```python
from ato.db_routers.sql.manager import SQLFinder

finder = SQLFinder(config)

# Find runs with same config structure
similar_runs = finder.find_similar_runs(run_id=123)
print(f"Found {len(similar_runs)} runs with same config structure")

# Check code version history
stats = finder.get_trace_statistics('my_project', trace_id='train_step')
print(f"Code versions: {stats['static_trace_versions']}")
print(f"Output versions: {stats['runtime_trace_versions']}")

# Find best run with specific code version
best_run = finder.find_best_run(
    project_name='my_project',
    metric_key='val_accuracy',
    mode='max'
)
print(f"Best accuracy: {best_run.metrics[-1].value}")
print(f"Code fingerprint: {best_run.fingerprints['train_step']}")
```

**Real-world scenario:**

You run 100 experiments. Result at epoch 50 suddenly jumps. Here's how to debug it:

```python
# Find when code changed
stats = finder.get_trace_statistics('my_project', trace_id='train_step')
# "3 different code versions across 100 runs"

# Find which runs used which version
runs = finder.get_runs_in_project('my_project')
by_code_version = {}
for run in runs:
    code_hash = run.fingerprints.get('train_step')
    by_code_version.setdefault(code_hash, []).append(run)

# Compare performance by code version
for code_hash, runs in by_code_version.items():
    avg_acc = mean([r.metrics[-1].value for r in runs])
    print(f"Code v{code_hash[:8]}: {avg_acc:.2%} avg accuracy")
```

**Result:** Code version `abc123` performs 5% better than `def456`. You can now trace exactly which commit introduced the change and why performance improved.

#### How Tracing Works

**Static Tracing (`@scope.trace`):**

Generates a fingerprint of the function's **logic**, not its name or formatting:

```python
# These three functions have IDENTICAL fingerprints
@scope.trace(trace_id='train_step')
def train_v1(config):
    loss = model(data)
    return loss

@scope.trace(trace_id='train_step')
def train_v2(config):
    # Added comments
    loss = model(data)  # Compute loss
    return loss

@scope.trace(trace_id='train_step')
def completely_different_name(config):
    loss=model(data)  # Different whitespace
    return loss
```

All three produce the **same fingerprint** because the underlying logic is identical. Comments, whitespace, and function names are ignored.

**Why this matters:**
- Refactoring doesn't create "new" code versions
- Safe renaming — fingerprint tracks behavior, not syntax
- Detects actual logic changes, not cosmetic edits

**When fingerprints change:**

```python
@scope.trace(trace_id='train_step')
def train_v1(config):
    loss = model(data)
    return loss

@scope.trace(trace_id='train_step')
def train_v2(config):
    loss = model(data) * 2  # ← Logic changed!
    return loss
```

Now fingerprints differ — you've changed the actual computation.

**Runtime Tracing (`@scope.runtime_trace`):**

Tracks what the function **produces**, not what it does:

```python
import numpy as np

# Basic: Track full output
@scope.runtime_trace(trace_id='predictions')
def evaluate(model, data):
    return model.predict(data)

# With init_fn: Fix randomness for reproducibility
@scope.runtime_trace(
    trace_id='predictions',
    init_fn=lambda: np.random.seed(42)  # Initialize before execution
)
def evaluate_with_dropout(model, data):
    return model.predict(data)  # Now deterministic

# With inspect_fn: Track specific parts of output
@scope.runtime_trace(
    trace_id='predictions',
    inspect_fn=lambda preds: preds[:100]  # Only hash first 100 predictions
)
def evaluate_large_output(model, data):
    return model.predict(data)

# Advanced: Type-only checking (ignore values)
@scope.runtime_trace(
    trace_id='predictions',
    inspect_fn=lambda preds: type(preds).__name__  # Track output type only
)
def evaluate_structure(model, data):
    return model.predict(data)
```

**Parameters:**
- `init_fn`: Optional function called before execution (e.g., seed fixing, device setup)
- `inspect_fn`: Optional function to extract/filter what to track (e.g., first N items, specific fields, types only)

Even if code hasn't changed, if predictions differ, the runtime fingerprint changes.

#### Static vs Runtime Tracing

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

**Example: Catching refactoring bugs**

```python
# Original implementation
@scope.trace(trace_id='forward_pass')
def forward(model, x):
    out = model(x)
    return out

# Safe refactoring: Added comments, changed variable name, different whitespace
@scope.trace(trace_id='forward_pass')
def forward(model,x):
    # Forward pass through model
    result=model(x)  # No spaces
    return result
```

These have **the same fingerprint** because the underlying logic is identical — only cosmetic changes (comments, whitespace, variable names).

```python
# Unsafe refactoring: Logic changed
@scope.trace(trace_id='forward_pass')
def forward(model, x):
    features = model.backbone(x)  # Now calling backbone + head separately!
    logits = model.head(features)
    return logits
```

This has a **different fingerprint** — the logic changed. If you expected them to be equivalent but they have different fingerprints, you've caught a refactoring bug.

---

## SQL Tracker: Experiment Tracking

Lightweight experiment tracking using SQLite.

### Why SQL Tracker?

- **Zero Setup**: Just a SQLite file, no servers
- **Full History**: Track all runs, metrics, and artifacts
- **Smart Search**: Find similar experiments by config structure
- **Code Versioning**: Track code changes via fingerprints
- **Offline-first**: No network required, sync to cloud tracking later if needed

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

### Automatic Step Calculation

```python
hyperband = HyperBand(scope, search_spaces, halving_rate=0.3, num_min_samples=4)

max_steps = 100000
steps_per_generation = hyperband.compute_optimized_initial_training_steps(max_steps)
# Example output: [27, 88, 292, 972, 3240, 10800, 36000, 120000]

# Use in training
@hyperband.main
def train(config):
    generation = config.__num_halved__
    num_steps = steps_per_generation[generation]

    metric = train_for_n_steps(num_steps)
    return metric
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

### Extensible Design

Ato's hyperopt module is built for extensibility:

| Component | Purpose |
|-----------|---------|
| `GridSpaceMixIn` | Parameter sampling logic (reusable) |
| `HyperOpt` | Base optimization class |
| `DistributedMixIn` | Distributed training support (optional) |

**Example: Implement custom search algorithm**

```python
from ato.hyperopt.base import GridSpaceMixIn, HyperOpt

class RandomSearch(GridSpaceMixIn, HyperOpt):
    def main(self, func):
        # Reuse GridSpaceMixIn.prepare_distributions()
        configs = self.prepare_distributions(self.config, self.search_spaces)

        # Implement random sampling
        import random
        random.shuffle(configs)

        results = []
        for config in configs[:10]:  # Sample 10 random configs
            metric = func(config)
            results.append((config, metric))

        return max(results, key=lambda x: x[1])
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
git clone https://github.com/yourusername/ato.git
cd ato
pip install -e .
```

### Release Philosophy

**Every release passes 100+ unit tests.**
**No unchecked code. No silent failure.**

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

---

## Composability

Ato is designed to **compose** with existing tools, not replace them.

### Works Where Other Systems Require Ecosystems

**Config composition:**
- Import OpenMMLab configs: `config.load_mm_config('mmdet_configs/faster_rcnn.py')`
- Load Hydra-style hierarchies: `ADict.compose_hierarchy(root='configs', select={'model': 'resnet50'})`
- Mix with argparse: `Scope(use_external_parser=True)`

**Experiment tracking:**
- Track locally in SQLite (zero setup)
- Sync to MLflow/W&B when you need dashboards
- Or use both: local SQLite + cloud tracking

**Hyperparameter optimization:**
- Built-in Hyperband
- Or compose with Optuna/Ray Tune — Ato's configs work with any optimizer

### What Makes Ato Different

Not features. **Architectural decisions.**

1. **Three-dimensional reproducibility** — Config structure + code bytecode + runtime output. Most tools track configs. Ato tracks causality.

2. **Content-based versioning** — No timestamps. No git commits. Just SHA256 fingerprints of what ran. Reproducibility becomes queryable.

3. **Namespace isolation** — MultiScope gives each team its own priority system. No more `model_lr` vs `data_lr` prefixes.

4. **Explicit dependencies** — Config chaining (`chain_with`) makes prerequisites visible. No more forgetting to call `base_setup`.

5. **Debuggable merging** — The `manual` command shows exactly how configs merged. Config bugs become traceable.

These aren't plugin features. They're **how Ato is built**.

### When to Use Ato

**Use Ato when:**
- You want zero boilerplate config management
- You need to debug why a config value isn't what you expect
- You're working on multi-team projects with namespace conflicts
- You want local-first experiment tracking
- You're migrating between config/tracking systems

**Ato works alongside:**
- Hydra (config composition)
- MLflow/W&B (cloud tracking)
- Optuna/Ray Tune (advanced hyperparameter search)
- PyTorch/TensorFlow/JAX (any ML framework)

---

## Roadmap

Ato's design constraint is **structural neutrality** — adding capabilities without creating dependencies.

### Planned: Local Dashboard (Optional Module)

A lightweight HTML dashboard for teams that want visual exploration without committing to cloud platforms:

**What it adds:**
- Metric comparison & trends (read-only view of SQLite data)
- Run history & artifact browsing
- Config diff visualization
- Interactive hyperparameter analysis

**Design constraints:**
- No hard dependency — Ato core works 100% without the dashboard
- Separate process — doesn't block or modify runs
- Zero lock-in — delete it anytime, training code doesn't change
- Composable — use alongside MLflow/W&B

**Guiding principle:** Ato remains a set of **independent, composable tools** — not a platform you commit to.

---

## License

MIT License
