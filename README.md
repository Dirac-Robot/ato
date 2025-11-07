# Ato: A Tiny Orchestrator

**Configuration, experimentation, and hyperparameter optimization for Python.**

No runtime magic. No launcher. No platform.
Just Python modules you compose.

```bash
pip install ato
```

---

## Design Philosophy

Ato was built on three constraints:

1. **Visibility** — When configs merge from multiple sources, you should see **why** a value was set.
2. **Composability** — Each module (ADict, Scope, SQLTracker, HyperOpt) works independently. Use one, use all, or mix with other tools.
3. **Structural neutrality** — Ato is a layer, not a platform. It has no opinion on your stack.

This isn't minimalism for its own sake.
It's **structural restraint** — interfering only where necessary, staying out of the way everywhere else.

**What Ato provides:**
- **Config composition** with explicit priority and merge order debugging
- **Namespace isolation** for multi-team projects (MultiScope)
- **Experiment tracking** in local SQLite with zero setup
- **Hyperparameter search** via Hyperband (or compose with Optuna/Ray Tune)

**What Ato doesn't provide:**
- Web dashboards (use MLflow/W&B)
- Model registry (use MLflow)
- Dataset versioning (use DVC)
- Plugin marketplace

Ato is designed to work **between** tools, not replace them.

---

## Quick Start

### 30-Second Example

```python
from ato.scope import Scope

scope = Scope()

@scope.observe(default=True)
def config(cfg):
    cfg.lr = 0.001
    cfg.batch_size = 32
    cfg.model = 'resnet50'

@scope
def train(cfg):
    print(f"Training {cfg.model} with lr={cfg.lr}")
    # Your training code here

if __name__ == '__main__':
    train()  # python train.py
    # Override from CLI: python train.py lr=0.01 model=%resnet101%
```

**Key features:**
- `@scope.observe()` defines config sources
- `@scope` injects the merged config
- CLI overrides work automatically
- Priority-based merging (defaults → named configs → CLI → lazy evaluation)

---

## Table of Contents

- [ADict: Enhanced Dictionary](#adict-enhanced-dictionary)
- [Scope: Configuration Management](#scope-configuration-management)
  - [MultiScope: Namespace Isolation](#multiscope-namespace-isolation)
  - [Config Documentation & Debugging](#configuration-documentation--debugging)
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
def defaults(cfg):
    cfg.lr = 0.001
    cfg.epochs = 100

@scope.observe(priority=1)  # Applied after defaults
def high_lr(cfg):
    cfg.lr = 0.01

@scope.observe(priority=2)  # Applied last
def long_training(cfg):
    cfg.epochs = 300
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

### Lazy Evaluation

Sometimes you need configs that depend on other values set via CLI:

```python
@scope.observe()
def base_config(cfg):
    cfg.model = 'resnet50'
    cfg.dataset = 'imagenet'

@scope.observe(lazy=True)  # Evaluated AFTER CLI args
def computed_config(cfg):
    # Adjust based on dataset
    if cfg.dataset == 'imagenet':
        cfg.num_classes = 1000
        cfg.image_size = 224
    elif cfg.dataset == 'cifar10':
        cfg.num_classes = 10
        cfg.image_size = 32
```

```bash
python train.py dataset=%cifar10% computed_config
# Results in: num_classes=10, image_size=32
```

**Python 3.11+ Context Manager**:

```python
@scope.observe()
def my_config(cfg):
    cfg.model = 'resnet50'
    cfg.num_layers = 50

    with Scope.lazy():  # Evaluated after CLI
        if cfg.model == 'resnet101':
            cfg.num_layers = 101
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
def config(cfg):
    cfg.lr = 0.001
    cfg.batch_size = 32
    cfg.model = 'resnet50'

@scope.manual
def config_docs(cfg):
    cfg.lr = 'Learning rate for optimizer'
    cfg.batch_size = 'Number of samples per batch'
    cfg.model = 'Model architecture (resnet50, resnet101, etc.)'
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
def defaults(cfg):
    cfg.lr = 0.001

@scope.observe(priority=1)
def experiment_config(cfg):
    cfg.lr = 0.01

@scope.observe(priority=2)
def another_config(cfg):
    cfg.lr = 0.1

@scope.observe(lazy=True)
def adaptive_lr(cfg):
    if cfg.batch_size > 64:
        cfg.lr = cfg.lr * 2
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
def config(cfg):
    cfg.lr = 0.001
    cfg.batch_size = 32

@scope
def train(cfg):
    print(f"GPU: {cfg.gpu}, LR: {cfg.lr}")

if __name__ == '__main__':
    parser.parse_args()  # Merges argparse with scope
    train()
```

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
def defaults(cfg):
    # Data
    cfg.data = ADict(
        dataset='cifar10',
        batch_size=32,
        num_workers=4
    )

    # Model
    cfg.model = ADict(
        backbone='resnet50',
        pretrained=True
    )

    # Training
    cfg.train = ADict(
        lr=0.001,
        epochs=100,
        optimizer='adam'
    )

    # Experiment tracking
    cfg.experiment = ADict(
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
def train(cfg):
    # Setup experiment tracking
    logger = SQLLogger(cfg)
    run_id = logger.run(tags=[cfg.model.backbone, cfg.data.dataset])

    try:
        # Training loop
        for epoch in range(cfg.train.epochs):
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

- Python >= 3.7
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

### Quality Assurance

Ato's design philosophy — **structural neutrality** and **debuggable composition** — extends to our testing practices.

**Release Policy:**
- **All 100+ unit tests must pass before any release**
- No exceptions, no workarounds
- Tests cover every module: ADict, Scope, MultiScope, SQLTracker, HyperBand

**Why this matters:**
When you build on Ato, you're trusting it to stay out of your way. That means zero regressions, predictable behavior, and reliable APIs. Comprehensive test coverage ensures that each component works independently and composes correctly.

Run tests locally:
```bash
python -m pytest unit_tests/
```

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

### Three Capabilities Other Tools Don't Provide

1. **MultiScope** — True namespace isolation with independent priority systems
2. **`manual` command** — Visualize exact config merge order for debugging
3. **Structural hashing** — Track when experiment **architecture** changes, not just values

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
