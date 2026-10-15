# janet-xgboost

Typed xgboost bindings for Janet over the pinned C API, via core ffi -
no jpm native module, no C build. libxgboost 3.4.1 is pinned to the
official wheels (sha256-verified installer), so
predictions are bitwise-identical to the official python binding on
the same wheel - the official python binding is the external
cross-check, and the conformance is checked bitwise, not to a
tolerance. Two ingest routes
reach the train side: column-major Janet arrays, and the
[janet-num](https://github.com/jennystats/janet-num) packed f32
handle, which crosses to the library with zero Janet elements.

Part of [Jenny Stats](https://github.com/jennystats), a home for Janet
data and statistics packages. Jenny is just Janet.

## Import

Import: `(import janet-xgboost)`, call sites
`(janet-xgboost/predict ...)`. Shorter call sites via
`(import janet-xgboost :as xgboost)` - the form used in our tests
and benches.

## Example

```janet
(import janet-xgboost)

(def cols [[0.1 0.9 0.4 0.7] [0.2 0.8 0.5 0.1]])  # column-major
(def labels [1.0 4.0 2.0 3.0])

(with [dmx (janet-xgboost/dmatrix-from-columns cols labels)]
  (def bst (-> (janet-xgboost/booster @[dmx] {:objective "reg:squarederror"
                                              :seed 42})
               (janet-xgboost/train dmx 10)))
  (printf "%q" (janet-xgboost/predict bst dmx)))
# @[1.3011019229888916 3.6988980770111084 2.0694849491119385 2.9305150508880615]
```

The same fixture via the janet-num packed handle:

```janet
(import janet-num)
(import janet-xgboost :as xgboost)

(def packed (janet-num/pack-columns-f32 cols))  # {:ptr :nrow :ncol :free}
(with [dmx (xgboost/dmatrix-from-packed packed labels)]
  # train and predict as above; predictions bitwise-identical
  # to the columns route
  ...)
```

## API

Thirteen public functions; everything else in the module is internal.
`dmx-set-label`, `set-param` and `train` return their handle, so calls
compose left to right.

| function | what it does |
|---|---|
| `(version)` | libxgboost version triple, `[3 4 1]` on the pin |
| `(global-config json)` | set the library global config (set-only side) |
| `(get-global-config)` | read it back; the string is library-owned, valid until the next call |
| `(dmatrix-from-file path)` | dmatrix from a URI: `"<path>?format=libsvm"` or `?format=csv` |
| `(dmatrix-from-columns cols &opt labels)` | dmatrix from column-major arrays |
| `(dmatrix-from-packed packed &opt labels)` | dmatrix from a janet-num packed f32 handle, zero Janet elements |
| `(dmx-set-label dmx labels)` | set (or replace) the labels; chains |
| `(booster &opt dmats params)` | booster handle over the training dmatrix list |
| `(set-param bst k v)` | one xgboost param; chains |
| `(train bst dmx &opt rounds)` | run `rounds` boosting rounds (default 10); chains |
| `(predict bst dmx)` | flat array, or row arrays for multi-class |
| `(save-model bst path)` | `.json` or `.ubj`, chosen by the extension |
| `(load-model path &opt params)` | booster handle from a saved model |

Behavioural detail - preconditions, ownership, the type matrix, the
error taxonomy and the libxgboost version coupling - is the contract:
`specs/api-contract.md` (v1.3.1).

## Error classes

- **Library errors (the ok! class)**: every nonzero return code from a
  bound call raises `xgboost: <what> failed: <library diagnostic>` -
  one fixed `<what>` per call site, the library's own message carried
  through.
- **Type guards**: a handle argument that is not a struct, carries
  the wrong `:kind`, or carries a `:ptr` that is not a true pointer
  raises `expected dmatrix handle` / `expected booster handle` before
  any ffi call.
- **Typed signatures**: nil or table arguments are rejected by the
  typed signatures before the guards - `arg N expected :struct, got
  :nil`, naming the position and the expected type.

## Install

The module binds xgboost's C API over ffi, so it needs the pinned
shared library (official wheels, sha256-verified, idempotent; CUDA
compiled in with CPU fallback, so no GPU is required). Platforms:
Linux x86_64 and macOS (Apple silicon + Intel) from the official
wheels; Windows is not supported - run the stack under WSL2.

```bash
libs/install-xgboost.sh   # libxgboost 3.4.1 -> ~/.local/lib/xgboost/
```

Module dependencies:
[janet-num](https://github.com/jennystats/janet-num) (the packed-f32
ingest kernel),
[janet-kit](https://github.com/jennystats/janet-kit) (runtime type
contracts) and [janet-json](https://github.com/jennystats/janet-json).
With the library in place:

```bash
jpm install https://github.com/jennystats/janet-xgboost
```

Lib resolution at import: `$XGBOOST_LIB`, then
`~/.local/lib/xgboost/xgboost/lib/libxgboost.so` (Linux) or
`libxgboost.dylib` (macOS), then the system
dlopen search. The import fails with an actionable message if none
loads.

## Tests

```bash
./run_tests.sh
```

The smoke suite covers lifecycle, both ingest routes, training and
predict conformance against recorded exact predictions, the error
taxonomy, the global-config round trip and save/load. It skips
cleanly when libxgboost or the janet-num kernel is absent;
present-but-broken is a fail, never a skip.

## Known cuts, declared

- No training orchestration: no early stopping, no cross-validation,
  no custom objectives.
- One predict shape only (value output, all trees): margin,
  contribution and tree-range variants are not exposed.
- No CUDA surface, no python bridge, no CLI fallback.
- Closed handles are close-then-discard: use-after-close is undefined
  behaviour, unguarded by design.
- libxgboost is pinned to 3.4.1; no support is offered for other
  versions.

## Alternatives

- **[xgboost](https://github.com/dmlc/xgboost)** - the upstream project
  itself, with mature python, R and JVM bindings. For users outside
  the Janet ecosystem, or needing the full python API surface, it is
  the direct route.

## License

AGPLv3 for the module - see `LICENSE`. The bound libxgboost is
Apache-2.0, loaded at runtime; this is a fresh ffi binding, not a
port.

## AI assistance

The code in this repository was written with the assistance of a
large language model and reviewed by its maintainer.
