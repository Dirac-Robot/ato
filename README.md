# Ato

Ato is intentionally small — it’s not about lines of code,  
it’s about where they belong.  
The core fits in a few hundred lines because it doesn’t need to fight Python — it flows with it.

---

**Ato** is a lightweight Python library for experiment management in machine learning and data science.  
It provides flexible configuration management, experiment tracking, and hyperparameter optimization —  
all without the complexity or overhead of heavy frameworks.

## Why Ato?

### Core Differentiators

- **True Namespace Isolation**: MultiScope provides independent config contexts (unique to Ato!)
- **Configuration Transparency**: Visualize exact config merge order - debug configs with `manual` command
- **Built-in Experiment Tracking**: SQLite-based tracking with no external services required
- **Structural Hashing**: Track experiment structure changes automatically

### Developer Experience

- **Zero Boilerplate**: Auto-nested configs, lazy evaluation, attribute access
- **CLI-first Design**: Configure experiments from command line without touching code
- **Framework Agnostic**: Works with PyTorch, TensorFlow, JAX, or pure Python

## Quick Start

```bash
pip install ato
```

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

---

## Table of Contents

- [ADict: Enhanced Dictionary](#adict-enhanced-dictionary)
- [Scope: Configuration Management](#scope-configuration-management)
  - [MultiScope: Namespace Isolation](#2-multiscope---multiple-configuration-contexts) ⭐ Unique to Ato
  - [Config Documentation & Debugging](#5-configuration-documentation--inspection) ⭐ Unique to Ato
- [SQL Tracker: Experiment Tracking](#sql-tracker-experiment-tracking)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Best Practices](#best-practices)
- [Comparison with Hydra](#ato-vs-hydra)

---

## ADict: Enhanced Dictionary

`ADict` is an enhanced dictionary designed for managing experiment configurations. It combines the simplicity of Python dictionaries with powerful features for ML workflows.

### Core Features

These are the fundamental capabilities that make ADict powerful for experiment management:

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| **Structural Hashing** | Hash based on keys + types, not values | Track when experiment structure changes |
| **Nested Access** | Dot notation for nested configs | `config.model.lr` instead of `config['model']['lr']` |
| **Format Agnostic** | Load/save JSON, YAML, TOML, XYZ | Work with any config format |
| **Safe Updates** | `update_if_absent()` method | Prevent accidental overwrites |

### Developer Convenience Features

These utilities maximize developer productivity and reduce boilerplate:

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Auto-nested (`ADict.auto()`)** | Infinite depth lazy creation | `config.a.b.c = 1` just works - no KeyError |
| **Attribute-style Assignment** | `config.lr = 0.1` | Cleaner, more readable code |
| **Conditional Updates** | Only update missing keys | Merge configs safely |

### Quick Examples

```python
from ato.adict import ADict

# Structural hashing - track config structure changes
config1 = ADict(lr=0.1, epochs=100, model='resnet50')
config2 = ADict(lr=0.01, epochs=200, model='resnet101')
print(config1.get_structural_hash() == config2.get_structural_hash())  # True

config3 = ADict(lr=0.1, epochs='100', model='resnet50')  # epochs is str!
print(config1.get_structural_hash() == config3.get_structural_hash())  # False

# Load/save any format
config = ADict.from_file('config.json')
config.dump('config.yaml')

# Safe updates
config.update_if_absent(lr=0.01, scheduler='cosine')  # Only adds scheduler
```

### Convenience Features in Detail

#### Auto-nested: Zero Boilerplate Config Building

The most loved feature - no more manual nesting:

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

**Perfect for Scope integration**:

```python
from ato.scope import Scope

scope = Scope()

@scope.observe(default=True)
def config(cfg):
    # No pre-definition needed!
    cfg.training.optimizer.name = 'AdamW'
    cfg.training.optimizer.lr = 0.001
    cfg.model.encoder.num_layers = 12
```

**Works with CLI**:

```bash
python train.py model.backbone.resnet.depth=50 data.batch_size=32
```

#### More Convenience Utilities

```python
# Attribute-style access
config.lr = 0.1
print(config.lr)  # Instead of config['lr']

# Nested access
print(config.model.backbone.type)  # Clean and readable

# Conditional updates - merge configs safely
base_config.update_if_absent(**experiment_config)
```

---

## Scope: Configuration Management

Scope solves configuration complexity through **priority-based merging** and **CLI integration**. No more scattered config files or hard-coded parameters.

### Key Concepts

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

### Advanced Features

#### 1. Lazy Evaluation - Dynamic Configuration

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

#### 2. MultiScope - Multiple Configuration Contexts

**Unique to Ato**: Manage completely separate configuration namespaces. Unlike Hydra's config groups, MultiScope provides true **namespace isolation** with independent priority systems.

##### Why MultiScope?

| Challenge | Hydra's Approach | Ato's MultiScope |
|-----------|------------------|---------------------|
| Separate model/data configs | Config groups in one namespace | **Independent scopes with own priorities** |
| Avoid key collisions | Manual prefixing (`model.lr`, `train.lr`) | **Automatic namespace isolation** |
| Different teams/modules | Single config file | **Each scope can be owned separately** |
| Priority conflicts | Global priority system | **Per-scope priority system** |

##### Basic Usage

```python
from ato.scope import Scope, MultiScope

model_scope = Scope(name='model')
data_scope = Scope(name='data')
scope = MultiScope(model_scope, data_scope)

@model_scope.observe(default=True)
def model_config(model):
    model.backbone = 'resnet50'
    model.pretrained = True

@data_scope.observe(default=True)
def data_config(data):
    data.dataset = 'cifar10'
    data.batch_size = 32

@scope
def train(model, data):  # Named parameters match scope names
    print(f"Training {model.backbone} on {data.dataset}")
```

##### Real-world: Team Collaboration

Different team members can own different scopes without conflicts:

```python
# team_model.py - ML team owns this
model_scope = Scope(name='model')

@model_scope.observe(default=True)
def resnet_default(model):
    model.backbone = 'resnet50'
    model.lr = 0.1  # Model-specific learning rate

@model_scope.observe(priority=1)
def resnet101(model):
    model.backbone = 'resnet101'
    model.lr = 0.05  # Different lr for bigger model

# team_data.py - Data team owns this
data_scope = Scope(name='data')

@data_scope.observe(default=True)
def cifar_default(data):
    data.dataset = 'cifar10'
    data.lr = 0.001  # Data augmentation learning rate (no conflict!)

@data_scope.observe(priority=1)
def imagenet(data):
    data.dataset = 'imagenet'
    data.workers = 16

# train.py - Integration point
from team_model import model_scope
from team_data import data_scope

scope = MultiScope(model_scope, data_scope)

@scope
def train(model, data):
    # Both have 'lr' but in separate namespaces!
    print(f"Model LR: {model.lr}, Data LR: {data.lr}")
```

**Key advantage**: `model.lr` and `data.lr` are completely independent. No need for naming conventions like `model_lr` vs `data_lr`.

##### CLI with MultiScope

Override each scope independently:

```bash
# Override model scope only
python train.py model.backbone=%resnet101%

# Override data scope only
python train.py data.dataset=%imagenet%

# Override both
python train.py model.backbone=%resnet101% data.dataset=%imagenet%

# Call named configs per scope
python train.py resnet101 imagenet
```

#### 3. Import/Export Configs

Ato supports importing configs from multiple frameworks:

```python
@scope.observe()
def load_external(config):
    # Load from any format
    config.load('experiments/baseline.json')
    config.load('models/resnet.yaml')

    # Export to any format
    config.dump('output/final_config.toml')

    # Import OpenMMLab configs - handles _base_ inheritance automatically
    config.load_mm_config('mmdet_configs/faster_rcnn.py')
```

**OpenMMLab compatibility** is built-in:
- Automatically resolves `_base_` inheritance chains
- Supports `_delete_` keys for config overriding
- Makes migration from MMDetection/MMSegmentation/etc. seamless

**Hydra-style config composition** is also built-in via `compose_hierarchy`:

```python
from ato.adict import ADict

# Hydra-style directory structure:
# configs/
#   ├── config.yaml          # base config
#   ├── model/
#   │   ├── resnet50.yaml
#   │   └── resnet101.yaml
#   └── data/
#       ├── cifar10.yaml
#       └── imagenet.yaml

config = ADict.compose_hierarchy(
    root='configs',
    config_filename='config',
    select={
        'model': 'resnet50',      # or ['resnet50', 'resnet101'] for multiple
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

**Key features**:
- Config groups (model/, data/, optimizer/, etc.)
- Automatic file discovery (tries .yaml, .json, .toml, .xyz)
- Dotted overrides (`model.lr=0.01`)
- Required key validation
- Flexible error handling

#### 4. Argparse Integration

Mix Ato with existing argparse code:

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

#### 5. Configuration Documentation & Inspection

**One of Ato's most powerful features**: Auto-generate documentation AND visualize the exact order of configuration application.

##### Basic Documentation

```python
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
defaults → (CLI Inputs) → lazy_config → main

(User Manuals)
config.lr: Learning rate for optimizer
config.batch_size: Number of samples per batch
config.model: Model architecture (resnet50, resnet101, etc.)
--------------------------------------------------
```

##### Why This Matters

The **applying order visualization** shows you **exactly** how your configs are merged:
- Which config functions are applied (in order)
- When CLI inputs override values
- Where lazy configs are evaluated
- The final function that uses the config

**This prevents configuration bugs** by making the merge order explicit and debuggable.

##### MultiScope Documentation

For complex projects with multiple scopes, `manual` shows each scope separately:

```python
from ato.scope import Scope, MultiScope

model_scope = Scope(name='model')
train_scope = Scope(name='train')
scope = MultiScope(model_scope, train_scope)

@model_scope.observe(default=True)
def model_defaults(model):
    model.backbone = 'resnet50'
    model.num_layers = 50

@model_scope.observe(priority=1)
def model_advanced(model):
    model.pretrained = True

@model_scope.observe(lazy=True)
def model_lazy(model):
    if model.backbone == 'resnet101':
        model.num_layers = 101

@train_scope.observe(default=True)
def train_defaults(train):
    train.lr = 0.001
    train.epochs = 100

@model_scope.manual
def model_docs(model):
    model.backbone = 'Model backbone architecture'
    model.num_layers = 'Number of layers in the model'

@train_scope.manual
def train_docs(train):
    train.lr = 'Learning rate for optimizer'
    train.epochs = 'Total training epochs'

@scope
def main(model, train):
    print(f"Training {model.backbone} with lr={train.lr}")

if __name__ == '__main__':
    main()
```

```bash
python train.py manual
```

**Output:**
```
--------------------------------------------------
[Scope "model"]
(The Applying Order of Views)
model_defaults → model_advanced → (CLI Inputs) → model_lazy → main

(User Manuals)
model.backbone: Model backbone architecture
model.num_layers: Number of layers in the model
--------------------------------------------------
[Scope "train"]
(The Applying Order of Views)
train_defaults → (CLI Inputs) → main

(User Manuals)
train.lr: Learning rate for optimizer
train.epochs: Total training epochs
--------------------------------------------------
```

##### Real-world Example

This is especially valuable when debugging why a config value isn't what you expect:

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
defaults → experiment_config → another_config → (CLI Inputs) → adaptive_lr → main
```

Now it's **crystal clear** why `lr=0.1` (from `another_config`) and not `0.01`!

---

## SQL Tracker: Experiment Tracking

Lightweight experiment tracking using SQLite - no external services, no setup complexity.

### Why SQL Tracker?

- **Zero Setup**: Just a SQLite file, no servers
- **Full History**: Track all runs, metrics, and artifacts
- **Smart Search**: Find similar experiments by config structure
- **Code Versioning**: Track code changes via fingerprints

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

### Quick Start

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

### Real-world Example: Experiment Comparison

```python
# Compare hyperparameter impact
finder = SQLFinder(config)

runs = finder.get_runs_in_project('my_project')
for run in runs:
    # Get final accuracy
    final_metrics = [m for m in run.metrics if m.key == 'val_accuracy']
    best_acc = max(m.value for m in final_metrics) if final_metrics else 0

    print(f"LR: {run.config.lr}, Batch: {run.config.batch_size} → Acc: {best_acc:.2%}")
```

### Features Summary

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

### Extensible Design

Ato's hyperopt module is built for extensibility and reusability:

| Component | Purpose | Benefit |
|-----------|---------|---------|
| `GridSpaceMixIn` | Parameter sampling logic | Reusable across different algorithms |
| `HyperOpt` | Base optimization class | Easy to implement custom strategies |
| `DistributedMixIn` | Distributed training support | Optional, composable |

**This design makes it trivial to implement custom search algorithms**:

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

Let Hyperband compute optimal training steps:

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

### Distributed Hyperparameter Search

Ato supports distributed hyperparameter optimization out of the box:

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

**Key features**:
- Automatic work distribution across GPUs
- Synchronized config selection via `broadcast_object_from_root`
- Results aggregation with `all_gather_object`
- Compatible with PyTorch DDP, FSDP, DeepSpeed

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

## License

MIT License

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Setup

```bash
git clone https://github.com/yourusername/ato.git
cd ato
pip install -e .
```

---

## Comparison with Other Tools

| Feature | Ato | MLflow | W&B | Hydra |
|---------|--------|--------|-----|-------|
| **Core Features** |
| Zero setup | ✅ | ❌ | ❌ | ✅ |
| Offline-first | ✅ | Partial | ❌ | ✅ |
| Config priority system | ✅ Explicit | Partial (Tags) | Partial (Run params) | ✅ Override |
| **True namespace isolation** | **✅ MultiScope** | **❌** | **❌** | **❌ Config groups only** |
| **Config merge visualization** | **✅ `manual`** | **❌** | **❌** | **Partial (`--cfg` tree)** |
| Structural hashing | ✅ | ❌ | ❌ | ❌ |
| Built-in HyperOpt | ✅ Hyperband | ❌ | ✅ Sweeps | Plugins (Optuna) |
| CLI-first design | ✅ | ❌ | ❌ | ✅ |
| **Compatibility** |
| Framework agnostic | ✅ | ✅ | ✅ | ✅ |
| Distributed training | ✅ Native + DDP/FSDP⁽¹⁾ | ✅ | ✅ | ✅ |
| Distributed HyperOpt | ✅ `DistributedHyperBand` | ❌ | Partial | Plugins |
| Hydra-style composition | ✅ `compose_hierarchy` | N/A | N/A | Native |
| OpenMMLab configs | ✅ `load_mm_config` | ❌ | ❌ | ❌ |
| **Visualization & UI** |
| Web dashboard | 🔜 Planned | ✅ | ✅ | ❌ |
| Real-time metrics | 🔜 Planned | ✅ | ✅ | ❌ |
| Interactive plots | 🔜 Planned | ✅ | ✅ | ❌ |
| Metric comparison UI | 🔜 Planned | ✅ | ✅ | ❌ |
| **Advanced Features** |
| Model registry | 🔜 Planned | ✅ | ✅ | ❌ |
| Dataset versioning | 🔜 Planned | Partial | ✅ | ❌ |
| Team collaboration | ✅ MultiScope⁽²⁾ | ✅ Platform | ✅ Platform | ❌ |

⁽¹⁾ Native distributed hyperparameter optimization via `DistributedHyperBand`. Regular training is compatible with any distributed framework (DDP, FSDP, DeepSpeed) - just integrate logging, no special code needed.

⁽²⁾ Team collaboration via MultiScope: separate config ownership per team (e.g., Team A owns model scope, Team B owns data scope) without naming conflicts.

**Note on config compatibility**: Ato provides built-in support for other config frameworks:
- **Hydra-style composition**: `compose_hierarchy()` supports config groups, select, overrides - full compatibility
- **OpenMMLab configs**: `load_mm_config()` handles `_base_` inheritance and `_delete_` keys
- Migration from existing projects is seamless - just import your configs and go

### Ato vs. Hydra

While Hydra is excellent for config composition, Ato provides unique features:

| Aspect | Hydra | Ato |
|--------|-------|--------|
| **Namespace isolation** | Config groups share namespace | ✅ MultiScope with independent namespaces<br/>(no key collisions) |
| **Priority system** | Single global override system | ✅ Per-scope priority + lazy evaluation |
| **Config merge debugging** | Tree view (`--cfg`)<br/>Shows final config | ✅ `manual` command<br/>Shows merge order & execution flow |
| **Experiment tracking** | Requires external tools<br/>(MLflow/W&B) | ✅ Built-in SQL tracker |
| **Team workflow** | Single config file ownership | ✅ Separate scope ownership per team⁽³⁾ |

⁽³⁾ Example: Team A defines `model_scope`, Team B defines `data_scope`, both can use `model.lr` and `data.lr` without conflicts.

**Use Ato over Hydra when:**
- Multiple teams need independent config ownership (MultiScope)
- You want to avoid key collision issues (no manual prefixing needed)
- You need to debug why a config value was set (`manual` command)
- You want experiment tracking without adding MLflow/W&B
- You're migrating from OpenMMLab projects

**Use Hydra when:**
- You have very deep config hierarchies with complex inheritance
- You prefer YAML over Python
- You need the mature plugin ecosystem (Ray, Joblib, etc.)
- You don't need namespace isolation

**Why not both?**
- Ato has **built-in Hydra-style composition** via `compose_hierarchy()`
- You can use Hydra's directory structure and config groups directly in Ato
- Get MultiScope + experiment tracking + merge debugging on top of Hydra's composition
- Migration is literally just replacing `hydra.compose()` with `ADict.compose_hierarchy()`

**Ato is for you if:**
- You want lightweight, offline-first experiment tracking
- You need **true namespace isolation for team collaboration**
- **You want to debug config merge order visually** (unique to Ato!)
- You prefer simple Python over complex frameworks
- You want reproducibility without overhead