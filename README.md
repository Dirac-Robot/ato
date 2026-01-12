# Ato: A Scope-Based Config Layer for ML

> **Describe experiments as composable config views. Get reproducibility for free.**

Ato is a **scope-based configuration layer** for ML experiments. It treats configuration not as static files, but as a **sequence of transformations (reasoning)** with explicit priorities and dependencies.

```bash
pip install ato
```

---

## Quick Start

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

@scope
def train(config):
    print(f'Training {config.model} for {config.epochs} epochs with lr={config.lr}')

if __name__ == '__main__':
    train()
```

**Run it:**

```bash
python train.py                    # lr=1e-3, epochs=50
python train.py high_lr            # lr=3e-3, epochs=50
python train.py long_run           # lr=3e-3, epochs=200 (chain_with auto-applies high_lr)
python train.py lr=0.01            # CLI override
python train.py manual             # Show view application order + docs
```

---

## Table of Contents

- [Core Concepts](#core-concepts)
  - [Views and Priority](#views-and-priority)
  - [ADict: Structure-Aware Dict](#adict-structure-aware-dict)
  - [Config Chaining](#config-chaining)
  - [Lazy Evaluation](#lazy-evaluation)
  - [MultiScope: Namespace Isolation](#multiscope-namespace-isolation)
- [CLI Syntax](#cli-syntax)
- [Fingerprinting (Reproducibility)](#fingerprinting-reproducibility)
  - [Code Fingerprinting](#code-fingerprinting)
  - [Runtime Fingerprinting](#runtime-fingerprinting)
- [SQL Tracker (Local Experiment Tracking)](#sql-tracker-local-experiment-tracking)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [FAQ](#faq)

---

## Core Concepts

### Views and Priority

Views are functions that modify configuration. **Lower priority is applied first.**

```python
@scope.observe(priority=-1)   # Applied first
def base(config):
    config.lr = 0.001

@scope.observe(priority=0)    # Applied second
def mid(config):
    config.lr = 0.01

@scope.observe(priority=1)    # Applied last
def high(config):
    config.lr = 0.1
```

**Application order:**
```
Default Views (priority=0) → Named Views (by priority) → CLI Arguments → Lazy Views
```

CLI arguments always have the **highest priority**.

---

### ADict: Structure-Aware Dict

`ADict` is not just a dict—it's a **structure-aware config object**.

```python
from ato.adict import ADict

# Nested access
config = ADict()
config.model.backbone.layers = [64, 128, 256]  # Auto-creates nested structure

# Structural hashing (tracks structure, not values)
config1 = ADict(lr=0.1, epochs=100)
config2 = ADict(lr=0.01, epochs=200)
config1.get_structural_hash() == config2.get_structural_hash()  # True (same structure)

config3 = ADict(lr=0.1, epochs='100')  # epochs is str!
config1.get_structural_hash() == config3.get_structural_hash()  # False (different structure)

# Access tracking
config = ADict(lr=0.1, epochs=100, unused=999)
_ = config.lr
config.get_minimal_config()  # {'lr': 0.1} - only accessed keys

# Freeze/Defrost
config.freeze()    # Read-only
config.defrost()   # Editable

# File I/O
config = ADict.from_file('config.yaml')
config.dump('config.json')
```

---

### Config Chaining

Declare dependencies with `chain_with` to auto-apply prerequisite views.

```python
@scope.observe()
def base_setup(config):
    config.project_name = 'my_project'
    config.data_dir = '/data'

@scope.observe(chain_with='base_setup')  # base_setup applied first
def advanced_training(config):
    config.distributed = True

@scope.observe(chain_with=['base_setup', 'gpu_setup'])  # Multiple dependencies
def multi_node_training(config):
    config.nodes = 4
```

```bash
python train.py advanced_training
# Result: base_setup → advanced_training
```

---

### Lazy Evaluation

Views with `lazy=True` are executed **after** CLI arguments are applied.

```python
@scope.observe()
def base_config(config):
    config.dataset = 'imagenet'

@scope.observe(lazy=True)  # Executed after CLI args
def computed_config(config):
    if config.dataset == 'imagenet':
        config.num_classes = 1000
    elif config.dataset == 'cifar10':
        config.num_classes = 10
```

```bash
python train.py dataset:=cifar10 computed_config
# Result: num_classes=10
```

**Python 3.11+ Context Manager:**

```python
@scope.observe()
def my_config(config):
    config.model = 'resnet50'
    config.num_layers = 50

    with Scope.lazy():  # Executed after CLI
        if config.model == 'resnet101':
            config.num_layers = 101
```

---

### MultiScope: Namespace Isolation

Manage completely separate configuration namespaces.

```python
from ato.scope import Scope, MultiScope

model_scope = Scope(name='model')
data_scope = Scope(name='data')
scope = MultiScope(model_scope, data_scope)

@model_scope.observe(default=True)
def model_config(model):
    model.backbone = 'resnet50'
    model.lr = 0.1  # Model learning rate

@data_scope.observe(default=True)
def data_config(data):
    data.dataset = 'cifar10'
    data.lr = 0.001  # Data augmentation LR (no conflict!)

@scope
def train(model, data):  # Parameter names match scope names
    print(f'Model LR: {model.lr}, Data LR: {data.lr}')
```

```bash
python train.py model.backbone:=resnet101 data.dataset:=imagenet
```

---

## CLI Syntax

### Basic Syntax

| Type | Syntax | Example |
|------|--------|---------|
| **Apply view** | `view_name` | `python train.py high_lr long_run` |
| **Python expression** | `key=value` | `lr=0.01`, `layers=[1,2,3]`, `enable=True` |
| **String literal** | `key:=value` | `model:=resnet50`, `name:="Hello World"` |
| **Nested key** | `a.b.c=value` | `model.backbone:=resnet101` |

### String Assignment (`:=` Syntax)

Unlike Python expressions (`=`), `:=` assigns values **directly as strings**.

```bash
# Simple string without quotes
python train.py model:=resnet50

# String with spaces requires quotes
python train.py prompt:="Hello World"
python train.py prompt:='Hello World'

# Mixed usage
python train.py lr=0.01 layers=[1,2,3] name:=experiment_1
```

### MultiScope CLI

```bash
python train.py model.backbone:=resnet101 data.batch_size=64
```

---

## Fingerprinting (Reproducibility)

### Code Fingerprinting

Tracks **logic changes**, ignoring comments, whitespace, and variable names.

```python
@scope.trace(trace_id='train_step')
@scope
def train_v1(config):
    loss = model(data)
    return loss

@scope.trace(trace_id='train_step')
@scope
def train_v2(config):
    # Added comment
    loss = model(data)  # Same logic
    return loss

# train_v1 and train_v2 have identical fingerprints (same logic)
```

**When fingerprint changes:**

```python
@scope.trace(trace_id='train_step')
@scope
def train_v3(config):
    loss = model(data) * 2  # Logic changed!
    return loss
```

---

### Runtime Fingerprinting

Tracks **actual function outputs**.

```python
import numpy as np

# Basic: Track full output
@scope.runtime_trace(trace_id='predictions')
@scope
def evaluate(model, data):
    return model.predict(data)

# init_fn: Fix randomness
@scope.runtime_trace(
    trace_id='predictions',
    init_fn=lambda: np.random.seed(42)
)
@scope
def evaluate_with_dropout(model, data):
    return model.predict(data)

# inspect_fn: Track specific parts
@scope.runtime_trace(
    trace_id='predictions',
    inspect_fn=lambda preds: preds[:100]  # Only first 100
)
@scope
def evaluate_large_output(model, data):
    return model.predict(data)
```

**When to use:**
- **Code fingerprinting**: Track code changes, verify refactoring
- **Runtime fingerprinting**: Detect non-determinism, debug silent failures

---

## SQL Tracker (Local Experiment Tracking)

Lightweight experiment tracking with SQLite. No server required.

### Logging

```python
from ato.db_routers.sql.manager import SQLLogger
from ato.adict import ADict

config = ADict(
    experiment=ADict(
        project_name='image_classification',
        sql=ADict(db_path='sqlite:///experiments.db')
    ),
    lr=0.001,
    batch_size=32
)

logger = SQLLogger(config)
run_id = logger.run(tags=['baseline', 'resnet50'])

for epoch in range(100):
    loss = train_one_epoch()
    acc = validate()
    logger.log_metric('train_loss', loss, step=epoch)
    logger.log_metric('val_accuracy', acc, step=epoch)

logger.log_artifact(run_id, 'checkpoints/model_best.pt', data_type='model')
logger.finish(status='completed')
```

### Querying

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
```

---

## Hyperparameter Optimization

Built-in **Hyperband** algorithm with successive halving.

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
    print(f'Best config: {best_result.config}')
    print(f'Best metric: {best_result.metric}')
```

---

## FAQ

### Does Ato replace Hydra?

No. Hydra focuses on hierarchical composition and overrides; Ato focuses on priority-based reasoning and causality tracking. Use them together or separately.

### Does Ato conflict with MLflow/W&B?

No. MLflow/W&B provide dashboards and cloud tracking; Ato provides local causality tracking (config reasoning + code fingerprinting). Use them together: MLflow/W&B for metrics/dashboards, Ato for "why did this change?"

### Do I need a server?

No. Ato uses local SQLite. Zero setup, zero network calls.

### Can I use Ato with existing config files?

Yes. Ato is format-agnostic:
- Load YAML/JSON/TOML → Ato fingerprints the result
- Use argparse → Ato integrates seamlessly
- Import OpenMMLab configs → `_base_` inheritance handled

### What's the performance overhead?

Minimal:
- Config fingerprinting: microseconds
- Code fingerprinting: once at decoration time
- Runtime fingerprinting: depends on `inspect_fn` complexity
- SQLite logging: milliseconds per metric

---

## Requirements

- Python >= 3.7 (Lazy evaluation requires Python >= 3.8)
- SQLAlchemy (for SQL Tracker)
- PyYAML, toml (for config serialization)

See `pyproject.toml` for full dependencies.

---

## Contributing

Contributions are welcome! Submit issues or pull requests.

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
