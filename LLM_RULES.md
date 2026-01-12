# Ato Usage Rules (LLM Reference)

Ato is a **scope-based config library** for ML. Config = reasoning (sequence of prioritized views).

---

## Core Pattern

```python
from ato.scope import Scope

scope = Scope()

@scope.observe(default=True)      # Always applied (priority 0)
def defaults(config):
    config.lr = 1e-3
    config.epochs = 50

@scope.observe(priority=1)        # Higher priority = applied later
def high_lr(config):
    config.lr = 3e-3

@scope.observe(priority=2, chain_with='high_lr')  # Auto-applies dependency
def long_run(config):
    config.epochs = 200

@scope.observe(lazy=True)         # Evaluated AFTER CLI args
def adaptive(config):
    if config.epochs > 100:
        config.lr *= 0.5

@scope                            # Entrypoint
def train(config):
    print(f'lr={config.lr}, epochs={config.epochs}')

if __name__ == '__main__':
    train()
```

## Application Order
```
Default Views → Named Views (by priority) → CLI Args → Lazy Views
```

---

## CLI Syntax

| Purpose | Syntax | Example |
|:--------|:-------|:--------|
| Apply view | `view_name` | `high_lr`, `long_run` |
| Python expr | `key=value` | `lr=0.01`, `layers=[1,2,3]`, `use_gpu=True` |
| String literal | `key:=value` | `name:=exp1`, `path:="/data/dir"` |
| MultiScope | `scope.key=value` | `model.lr=0.01`, `data.path:=/data` |

```bash
python train.py                         # defaults
python train.py high_lr                  # apply view
python train.py high_lr long_run        # multiple views
python train.py lr=0.01                 # Python expression override
python train.py name:=experiment_1      # string literal (:=)
python train.py prompt:="Hello World"   # string with spaces
python train.py manual                   # show view order + docs
```

---

## ADict (Config Dict)

```python
from ato.adict import ADict

config = ADict(lr=0.1, model=ADict(layers=[64, 128]))
config.model.layers.append(256)              # Nested access
config = ADict.from_file('config.yaml')      # Load file
config.dump('config.json')                   # Save file
config.get_structural_hash()                 # Structure hash (not values)
config.freeze()                              # Read-only
config.defrost()                             # Editable
```

---

## MultiScope (Namespace Isolation)

```python
from ato.scope import Scope, MultiScope

model_scope = Scope(name='model')
data_scope = Scope(name='data')
scope = MultiScope(model_scope, data_scope)

@model_scope.observe(default=True)
def model_config(model):
    model.lr = 0.1

@data_scope.observe(default=True)
def data_config(data):
    data.lr = 0.001  # Independent from model.lr

@scope
def train(model, data):  # Param names = scope names
    print(f'{model.lr}, {data.lr}')
```

CLI: `python train.py model.lr=0.01 data.lr=0.001`

---

## Fingerprinting

```python
@scope.trace(trace_id='train_step')   # Code fingerprint (ignores comments/whitespace)
@scope
def train_epoch(config): ...

@scope.runtime_trace(                 # Output fingerprint
    trace_id='predictions',
    init_fn=lambda: np.random.seed(42),
    inspect_fn=lambda x: x[:100]
)
@scope
def evaluate(config): ...
```

---

## Quick Reference

| Decorator | Purpose |
|:----------|:--------|
| `@scope.observe(default=True)` | Default view (always applied) |
| `@scope.observe(priority=N)` | Priority N (lower = earlier) |
| `@scope.observe(chain_with='...')` | Auto-apply dependency views |
| `@scope.observe(lazy=True)` | Evaluate after CLI args |
| `@scope.manual` | Document config keys |
| `@scope.trace(trace_id='...')` | Code fingerprint |
| `@scope.runtime_trace(...)` | Output fingerprint |
| `@scope` | Entrypoint function |
