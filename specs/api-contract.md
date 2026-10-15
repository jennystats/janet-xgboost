# Spec - janet-xgboost public API contract (v1.3.1, libxgboost 3.4.1)

Prose contract (a formal specification language is not the right tool
here; the xgboost C API is the external truth). Canonical signatures: the module
docstrings in `janet-xgboost.janet`. This document adds what
signatures cannot carry: preconditions, effects, ownership, the type
matrix and the error taxonomy. Binding truth: `janet-xgboost.janet`
(ffi/defbind,
pinned to libxgboost 3.4.1 from
`libs/install-xgboost.sh`); external truth: the xgboost C API.

## 1. Layering

- **Janet wrapper layer** (janet-xgboost.janet): handle construction,
  parameter stringification, prediction reshaping, actionable errors.
  Minimal by design - the C API is small and stable, and the wrapper
  adds no validation the C layer does not already do better.
- **C API layer** (libxgboost 3.4.1): reached only through
  `ffi/defbind` bindings with verbatim XG symbol names. Handles are
  raw pointers - no struct layouts cross the boundary, so the
  coupling surface is symbol names and call signatures only.
- **Handle structs**: dmatrix and   booster handles are structs with
  `:ptr`, a `:kind` marker (`:dmatrix` / `:booster`) and an idempotent
  `:close`. `with` works through `:close` on both handle types. The
  two kinds are otherwise structurally identical; the marker (added
  v1.2) is what makes them distinguishable -
  every dmatrix/booster-taking fn checks it (section 5, the
  type-guard class).
- **janet-num dependency**: column-major ingest goes through
  janet-num's pack kernel (section 7). The module imports janet-num;
  xgboost without janet-num does not load.
- **Lib resolution** at import: `$XGBOOST_LIB` then
  `~/.local/lib/xgboost/xgboost/lib/libxgboost.so` (the wheel's
  auditwheel layout) then the system dlopen search. The load is probed
  before the ffi context is set; every failing path raises the same
  actionable error naming `libs/install-xgboost.sh`.
- What this deliberately is not: no training orchestration (no early
  stopping, no cross-validation, no custom objectives), no python
  bridge, no CUDA surface, no CLI fallback (no xgboost binary is
  assumed to exist).

## 2. Call shapes

Import style (canonical, every doc and example copies this form;
`janet-xgboost` is the house module name):

```
(import janet-xgboost)              # long form: calls are
                                    # janet-xgboost/predict, ...
(import janet-xgboost :as xgb)      # the :as alias is the user-side
                                    # option (the form the examples
                                    # below use)
```

Public surface (everything else in the module is internal):

```
(version)                          -> [major minor patch]  ([3 4 1] on the pin)
(global-config json)               -> nil   (set-only side of the global config)
(get-global-config)                -> json string  (library-owned,
                                      valid until the next call)
(dmatrix-from-file path)           -> dmatrix handle   (path is a URI:
                                      "<path>?format=libsvm" or "?format=csv")
(dmatrix-from-columns cols &opt labels) -> dmatrix handle {:ptr :nrow :ncol :close}
                                      (cols: array of column-major arrays)
(dmatrix-from-packed packed &opt labels) -> dmatrix handle
                                      (packed: janet-num/duckdb f32 handle)
(dmx-set-label dmx labels)          -> dmx  (chain)
(booster &opt dmats params)         -> booster handle
                                      (dmats: array of dmatrix handles, default
                                      []; a nil dmats is accepted as the
                                      empty list - documented acceptance,
                                      not an accident;
                                      params: struct, default {}).
                                      TRAIN wants the training dmatrix in
                                      the create list: a booster created
                                      with an empty dmats list raises the
                                      library's own `0 feature is supplied.
                                      Are you using raw Booster interface?`
                                      at train (upstream behaviour,
                                      identical in the python binding)
(set-param bst k v)                 -> bst  (chain)
(train bst dmx &opt rounds)         -> bst  (chain; rounds default 10)
(predict bst dmx)                   -> dim 1: flat array of floats
                                      dim 2: array of row arrays (multi-class)
(save-model bst path)               -> path
(load-model path &opt params)        -> booster handle
```

- Chain style: `dmx-set-label`, `set-param`, `train` return their
  handle so calls compose left to right.
- `booster` and `load-model` apply a params struct entry by entry
  (`eachp [k v] params`); keys and values are stringified - numbers
  may be given as numbers (`:seed 42` and `:seed "42"` are
  equivalent). Param names are xgboost's own (`:objective`,
  `:max_depth`, `:eta`, `:seed`, `:nthread`, ...). On 3.4.1
  `XGBoosterSetParam` accepts unknown keys SILENTLY - validation
  happens at use, not at set; a malformed VALUE surfaces at train
  with the library's own diagnostic through the module's error class
  (the v1.1 row claimed unknown names raise - wrong; corrected in
  v1.2). `nthread` is a booster param and also an
  accepted global-config key on 3.4.1 (it round-trips through
  get-global-config; unknown global keys DO raise).
- `train` runs `UpdateOneIter` rounds times, iteration counter 0-based;
  `rounds` 0 is a legal no-op returning the booster unchanged.
- `predict` always uses the module's fixed predict config (value
  output, all trees, no strict shape); per-call predict variants
  (margin, contribution, tree ranges) are deliberately not exposed.
- `save-model`/`load-model`: the file extension selects the format -
  `.json` saves json, `.ubj` saves the universal binary format.
- The URI convention for `dmatrix-from-file` is caller-burden by
  design: the format is the caller's choice, the module passes the
  path through verbatim into the create config (json escaping is the
  module's). A MISMATCHED format parameter (e.g. csv on a libsvm
  file)   can yield a valid but wrong-shaped dmatrix with no error -
  check `:nrow`/`:ncol` when the format is in doubt. An empty or
  whitespace-only URI raises the wrapper's
  type-guard class BEFORE the ffi call: the empty string segfaults
  the C library outright - an upstream libxgboost defect, reproduced
  identically by the official python binding; the
  wrapper converts it into the error taxonomy rather than passing
  the crash through.

## 3. Handle lifecycle

- **Raw-pointer discipline - a deliberate design point.** Frees call
  the C API with the handle value directly: `XGDMatrixFree(ptr)`,
  `XGBoosterFree(ptr)`. This is the opposite of the duckdb module's
  ptr-ptr discipline, and each mirrors its library: xgboost's C API
  frees by value (the prototype takes the handle itself), duckdb's
  destroy functions take the handle cell's address so they can null
  it. A uniform wrapper-side discipline would fight one of the two
  libraries; each wrapper follows its own.
- **Ownership**: the wrapper owns the C object. `:close` frees it
  exactly once - a closure-captured flag makes every `:close` after
  the first a no-op, so double-close is safe on both handle types.
- **After close**: no use-after-close guard exists. A closed handle's
  `:ptr` is a freed pointer (the `:kind` marker survives the close) -
  using it in any call that reads `:ptr` (predict, train, label,
  booster construction over a dmats list) is undefined behaviour
  (the ffi call on a dead pointer). The user contract is
  close-then-discard; discard-then-gc without close is a leak
  (xgboost exports no finaliser hook the wrapper can reach).
- **`with`**: both handle kinds work with `with` via `:close`.
- **Shape keys**: `dmatrix-from-columns` and `dmatrix-from-packed`
  handles carry `:nrow`/`:ncol`, from-file included (queried via the
  row/col bindings at create).

## 4. Type matrix

| crossing | janet in | C out/in | shape / error |
|---|---|---|---|
| features (columns) | array of column-major arrays of numbers | row-major f32 buffer via janet-num pack kernel | ragged or empty input raises in the pack kernel ("no columns", "ragged" - janet-side guards, attributed there); doubles narrow to f32 at the write |
| features (packed) | `{:ptr :nrow :ncol :free}` f32 handle | same buffer, zero janet crossing | an invalid packed handle is the janet-num contract's error class |
| features (file) | URI string | json config string | missing file or bad format parameter raises the create-failure message carrying the uri text; an empty/whitespace URI raises the wrapper guard before the ffi call (upstream segfault); a mismatched format can yield a wrong-shaped dmatrix silently - check `:nrow`/`:ncol` |
| missing values | any NaN in feature cells | `missing` = NaN at create | NaN cells are accepted as missing by xgboost's own semantics; no wrapper error |
| labels | array of numbers, length = nrow | f32 buffer, field "label" | a length mismatch raises in the C layer's set-info call; labels given at create and via `dmx-set-label` are the same path |
| params | struct, stringified k/v | name/value strings | unknown parameter names are accepted silently on 3.4.1 - validation happens at use, not at set; a malformed value surfaces at train with the library's diagnostic |
| global config | json string | string | malformed json raises the set-failure message |
| predictions (1-dim) | - | flat f32 array | flat array, one float per row (regression, binary logit) |
| predictions (2-dim) | - | rows x cols f32 array | one array per row, each of `num_class` entries (multi-class softprob); rows sum to 1 per softmax |
| predictions (0 rows) | 0-row dmatrix (both ingest routes construct) | rc 0, null out arrays | empty array `@[]` - matching the official python binding; the library logs an "Empty dataset" warning on stderr |
| version | - | three int out-cells | tuple of three ints; the call returns void in 3.x, no rc to check |

Floats cross as f32 everywhere (the C API's element type): doubles
narrow at the ffi/write boundary, which validates element types and
integrality natively (a non-number raises naming the element index).
Model files: `.json` (json) or `.ubj` (universal binary).

## 5. Error taxonomy

| class | source | message shape |
|---|---|---|
| XG | any nonzero rc from a bound call | `xgboost: <what> failed: <XGBGetLastError text>` - one fixed `<what>` per call site ("create dmatrix from mat", "train iter", "predict", ...); the rc itself is not re-exposed (the C API's XGBGetLastError has no code form) |
| TYPE-GUARD (janet-side) | a dmatrix/booster handle argument that is not a struct, carries the wrong `:kind`, or carries a `:ptr` that is not `:pointer`-typed (a forged struct; covers every non-pointer value, numeric and boolean included, which pre-v1.3 reached the native ffi slot message); booster dmats elements included; a packed handle whose `:ptr` is neither a buffer nor a true pointer in from-packed; an empty/whitespace from-file URI. Split with the typed signatures: a `nil` (or table) argument to a `defn/typed` public fn is rejected by the typed signature FIRST (`<fn>: arg N expected :struct, got :nil`) - that message is what a non-struct argument gets; the TYPE-GUARD message is   what a wrong-KIND or wrong-`:ptr` struct gets (v1.3) | `xgboost: expected <kind> handle, got <type>` / `xgboost: expected packed f32 handle, got <type>` / `xgboost: dmatrix-from-file requires a non-empty path URI` - raised BEFORE any ffi call; the forged-handle and empty-URI crash classes convert here, never to a C-level segfault |
| VALIDATION (janet-side) | column ingest | raised in the janet-num pack kernel: "no columns", "ragged" - attributed to janet-num, not this module |
| VALIDATION (native) | bad element types at ffi/write | janet's cfunction message naming the offending element index |
| ENVIRONMENT | libxgboost absent or unloadable | `xgboost: cannot load <lib> - run libs/install-xgboost.sh or set $XGBOOST_LIB` |

Every message is actionable: it names the failing operation, the
library's own diagnostic, or the installer. Errors are never
swallowed; there is no silent-success path.

## 6. Version coupling

libxgboost 3.4.1 is pinned by `libs/install-xgboost.sh` (the official
universal manylinux wheel, sha256 verified; CUDA compiled in with CPU
fallback, so no GPU is required; the wheel's libgomp layout must be
preserved). The module's 17 bindings were verified live on both the
2.1.4 and 3.4.1 generations (22 probe stages each); the surface is
stable across that span except:

- `XGDMatrixCreateFromURI` (3.x) replaced `XGDMatrixCreateFromFile`
  (2.x) - which already required the URI form in practice - so this
  binding is 3.x-coupled.
- `XGBoostVersion` returns void in 3.x (int in 2.x); the module never
  read the rc, so the binding is unaffected.
- `XGBGetGlobalConfig` exists in 3.4.1 and not in 2.1.4 - the get side
  of the global config is 3.4.1-coupled and fine with that under the
  pin.
- The predict config keys and both free functions are identical across
  the generations.

Conformance history: recorded fixture predictions carried unchanged
from the 2.1.4 wheel to 3.4.1 (regression-tree maths is
generation-stable for the fixture), and janet-vs-python prediction
identity is bitwise on 3.4.1 (same wheel artifact). A
future libxgboost generation bump requires: re-probing the bound symbol
set and signatures; re-running the conformance predictions against
the recorded values; re-running the python-bitwise cross-check as the
external cross-check. No support is offered for other libxgboost versions.

## 7. janet-num dependency

Two ingest routes, one kernel:

- **Columns route**: `(dmatrix-from-columns cols labels)` - janet-num's
  `pack-columns-f32` transposes column-major arrays into the
  row-major f32 buffer xgboost wants, in one C pass (the interpreted
  transpose it replaced was 13x slower). This is the
  duckdb-`columns`-feed shape.
- **Packed route**: `(dmatrix-from-packed packed labels)` - a
  `pack-columns-f32` result handle (from duckdb's query packing or
  janet-num directly) crosses with zero janet elements; NULLs arrive
  as NaN, which is xgboost's missing semantics. Conformance is
  bitwise-identical to the columns route. LIFETIME: the packed
  handle must outlive the dmatrix - xgboost copies at create in name
  only (the quantized DMatrix does not hold the buffer, but a freed
  pack handle passed to from-packed is use-after-free with zero
  signal; close-then-discard applies to the janet-num handle too -
  cross-module note, janet-num's guard business).
- The packed guard is STRUCTURAL, not kind-typed: it accepts a buffer
  or a true pointer `:ptr` plus the shape keys, and deliberately does
  not require a `:kind :packed` marker (janet-num pack handles carry
  none - adding one would be a janet-num change, ruled out by the
  smallest-safe-change call for v1.3). A LIVE dmatrix passed as packed
  is therefore caller error at the seam: its `:ptr` is a true pointer,
  passes the structural guard, and the dmatrix constructs SILENTLY
  (rc 0, shape taken from the struct's own keys). No crash and no
  memory unsafety occur. The library diagnostic, when it comes,
  appears at train ("Invalid shape of labels") and only on the
  label-absent path; with labels present the train runs silently on
  reinterpreted object bytes, and predict on such a handle returns
  unreliable numbers with no error of any kind (v1.3.1; documented
  boundary, not a guard class).
- The canonical pipeline (integration-demo): duckdb feature
  engineering to columns -> dmatrix -> train -> predict, one process,
  no CSV materialisation; the resident-booster batch-predict pattern
  is the cheap stage (per-call overhead is between the julia and
  python bindings).

## 8. Change protocol

A new public function or option is a minor version bump of this
contract plus a smoke row. A change to any signature, precondition,
type-matrix cell or error class in this document is breaking: major
bump, smoke updates, consumers re-verified, one commit.

Changes:

- **v1.3.1 (2026-10-16)**: the section 7 packed-seam boundary
  corrected to the observed behaviour: the v1.3 sentence promised a
  downstream error that fires only on the label-absent train path -
  from-packed over a live dmatrix constructs silently, the
  "Invalid shape of labels" diagnostic is train-only, a labelled
  train corrupts silently, and predict never errors (unreliable
  output instead). Section 2 booster row gains the empty-dmats train
  sentence (0-feature diagnostic, training wants the dmatrix in the
  create list). Doc-only.
- **v1.3 (2026-10-16)**: the
  TYPE-GUARD row gains the pointer-ness requirement - a true
  `:pointer`-typed `:ptr`, closing the forged-struct crash and
  silent-corruption classes -
  and the typed-signature/TYPE-GUARD split clause for
  nil/table arguments. The packed guard is documented
  structural (buffer-or-pointer + shape, no kind marker - the
  live-dmatrix-as-packed boundary). The booster row documents
  nil dmats as the empty list. Section 6 binding count
  corrected to 17 (verified by count - the v1.2 figure of 15
  was a stale census, no binding has landed since v1.0). Guard-
  tightening and documentation only; no signature
  change - minor bump.
- **v1.2 (2026-10-16)**: handle `:kind` markers
  + the TYPE-GUARD error class on every dmatrix/
  booster-taking fn; from-file empty-URI guard with the upstream-
  segfault attribution; 0-row predict returns the official python
  binding's empty array; params row corrected (unknown keys accepted
  silently on 3.4.1, malformed values surface at train);
  nthread is also a global-config key on 3.4.1; the csv
  wrong-shape caveat; the packed-handle lifetime note
  (v1.3 section 7); use-after-close wording widened to every call
  that reads `:ptr`. Additive and behaviour-repairing; no signature
  change - minor bump.
- **v1.1 (2026-10-16)**: the module rename
  (`janet-xgboost`), get-global-config (library-owned string,
  valid until the next call) and the from-file shape keys (queried at
  create, all dmatrix handles shaped) are live. The get-global-config
  row documents the string's lifetime. No behavioural change against
  v1.0's intent; doc-only.
- **v1.0 (2026-10-16)**: first written contract, from the as-built
  surface, with the get side of the global config, the from-file shape
  keys and the module rename specified ahead of their code.
