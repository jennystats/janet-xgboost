# janet-xgboost.janet - in-process xgboost for janet via core ffi.
#
# Public package: janet-xgboost. License: AGPLv3 (see LICENSE). The
# bound libxgboost is Apache-2.0 (no licence conflict; fresh ffi
# binding, not a port).
#
# Pinned C API: xgboost 3.4.1 (official universal wheel artifact - CUDA
# compiled in, CPU fallback; installer libs/install-xgboost.sh). The
# 2.1.4 -> 3.4.1 generation bump changed nothing in the bound surface
# except XGBoostVersion returning void and CreateFromURI replacing
# CreateFromFile (verified live, both generations, 22/22 probe stages
# each).
#
# Lib resolution: $XGBOOST_LIB -> ~/.local/lib/xgboost/xgboost/lib/
# libxgboost.so -> dlopen search. Import fails actionably if none loads.
#
# Public API (thin, stable; behavioural spec: the api-contract):
#   (version)                            -> [major minor patch]
#   (global-config json)                 -> nil   (set side)
#   (get-global-config)                  -> json string (3.4.1 C API;
#                                          library-owned, valid until
#                                          the next call)
#   (dmatrix-from-file path)             -> dmx  {:ptr :nrow :ncol :close}
#                                          path is a uri: "<path>?format=libsvm"
#   (dmatrix-from-columns cols &opt labels) -> dmx (column-major arrays)
#   (dmatrix-from-packed packed &opt labels) -> dmx (janet-num f32 handle)
#   (dmx-set-label dmx labels)           -> dmx
#   (booster &opt dmats params)           -> bst  {:ptr :close}
#   (set-param bst k v)                  -> bst
#   (train bst dmx &opt rounds)          -> bst   (UpdateOneIter x rounds)
#   (predict bst dmx)                     -> flat array | array of row arrays
#   (save-model bst path)                 -> path
#   (load-model path &opt params)         -> bst
# Handles close via (:close h); `with` works. Frees take the handle
# DIRECTLY (the C API frees by value, unlike duckdb's ptr-ptr destroy).
# Handles carry a :kind marker (:dmatrix/:booster); dmatrix/booster-
# taking fns raise the type-guard error class on a wrong-kind or
# missing-:ptr argument (never a C-level segfault; the v1.2 guards).

(import typed :prefix "")
(import json)
(import janet-num) # the ingest transpose runs in the janet-num C kernel

#==------------------------------------------------------------------------==#
# Lib resolution
#==------------------------------------------------------------------------==#

(defn/typed lib-path {:args [] :ret :string} []
  "Resolve libxgboost.so: $XGBOOST_LIB, then the pinned home path,
  then the bare soname for the system dlopen search."
  (def env (get (os/environ) "XGBOOST_LIB"))
  (if env env
    (let [home (get (os/environ) "HOME")
          pinned (string home "/.local/lib/xgboost/xgboost/lib/libxgboost.so")]
      (if (not (nil? (os/stat pinned))) pinned "libxgboost.so"))))

(def lib (lib-path))
(ffi/context lib)

(unless (try (do (ffi/native lib) true) ([_] false))
  (errorf "xgboost: cannot load %s - run libs/install-xgboost.sh or set $XGBOOST_LIB" lib))

#==------------------------------------------------------------------------==#
# Bindings (c_api.h, xgboost 3.4.1)
#==------------------------------------------------------------------------==#

(ffi/defbind XGBoostVersion :void [maj :ptr min :ptr pat :ptr])
(ffi/defbind XGBSetGlobalConfig :int [json :string])
(ffi/defbind XGBGetGlobalConfig :int [out :ptr])
(ffi/defbind XGBGetLastError :string [])
(ffi/defbind XGDMatrixCreateFromURI :int [config :string out :ptr])
(ffi/defbind XGDMatrixCreateFromMat :int [data :ptr nrow :uint64 ncol :uint64 missing :float out :ptr])
(ffi/defbind XGDMatrixSetFloatInfo :int [dmx :ptr field :string arr :ptr len :uint64])
(ffi/defbind XGDMatrixNumRow :int [dmx :ptr out :ptr])
(ffi/defbind XGDMatrixNumCol :int [dmx :ptr out :ptr])
(ffi/defbind XGBoosterCreate :int [dmats :ptr len :uint64 out :ptr])
(ffi/defbind XGBoosterSetParam :int [bst :ptr name :string value :string])
(ffi/defbind XGBoosterUpdateOneIter :int [bst :ptr iter :int dtrain :ptr])
(ffi/defbind XGBoosterPredictFromDMatrix :int [bst :ptr dmx :ptr config :string out-shape :ptr out-dim :ptr out-result :ptr])
(ffi/defbind XGBoosterSaveModel :int [bst :ptr fname :string])
(ffi/defbind XGBoosterLoadModel :int [bst :ptr fname :string])
(ffi/defbind XGBoosterFree :int [bst :ptr])
(ffi/defbind XGDMatrixFree :int [dmx :ptr])

(def PREDICT-CONFIG "{\"type\":0,\"training\":false,\"iteration_begin\":0,\"iteration_end\":0,\"strict_shape\":false}")

(defn/typed ok! {:args [:number :string] :ret :nil} [rc what]
  "Raise with the library's own last-error text unless the call
  returned 0. The XG error class; the type-guard class lives in
  check-handle (contract section 5)."
  (unless (zero? rc)
    (errorf "xgboost: %s failed: %s" what (XGBGetLastError))))

(defn/typed check-handle {:args [:any :keyword] :ret :any} [x kind]
  "Type-guard error class: every dmatrix/booster-taking fn raises on
  a wrong-kind, missing-:ptr or non-pointer-:ptr argument (presence
  alone let forged structs through to a crash or a silent garbage
  handle before v1.3; a real handle's :ptr is
  :pointer-typed)."
  (unless (and (or (struct? x) (table? x))
               (= :pointer (type (get x :ptr)))
               (= kind (get x :kind)))
    (errorf "xgboost: expected %s handle, got %s" kind (type x))))

(defn/typed dmx-free {:args [:any] :ret :any} [ptr]
  "XGDMatrixFree - frees by handle value (the raw-pointer
  discipline); returns the library rc, unchecked like every free."
  (XGDMatrixFree ptr))

(defn/typed make-dmx {:args [:any :any :any] :ret :struct} [ptr nrow ncol]
  "A dmatrix handle: raw pointer, shape keys, idempotent :close."
  (var closed false)
  {:ptr ptr :nrow nrow :ncol ncol :kind :dmatrix
   :close (fn [self] (unless closed (dmx-free (self :ptr)) (set closed true)) nil)})

(defn/typed make-bst {:args [:any] :ret :struct} [ptr]
  "A booster handle: raw pointer, idempotent :close."
  (var closed false)
  {:ptr ptr :kind :booster
   :close (fn [self] (unless closed (XGBoosterFree (self :ptr)) (set closed true)) nil)})

(defn/typed dmx-shape {:args [:any] :ret :tuple} [ptr]
  "Query [nrow ncol] from a live dmatrix (the from-file shape keys)."
  (def nr (ffi/write :uint64 0))
  (def nc (ffi/write :uint64 0))
  (ok! (XGDMatrixNumRow ptr nr) "dmatrix num row")
  (ok! (XGDMatrixNumCol ptr nc) "dmatrix num col")
  [(int/to-number (ffi/read :uint64 nr)) (int/to-number (ffi/read :uint64 nc))])

#==------------------------------------------------------------------------==#
# Public API
#==------------------------------------------------------------------------==#

(defn/typed version {:args [] :ret :tuple} []
  "The linked libxgboost version as [major minor patch] (the smoke
  pins [3 4 1] on the installer's wheel)."
  (def maj (ffi/write :int 0))
  (def min (ffi/write :int 0))
  (def pat (ffi/write :int 0))
  (XGBoostVersion maj min pat)
  [(ffi/read :int maj) (ffi/read :int min) (ffi/read :int pat)])

(defn/typed global-config {:args [:string] :ret :nil} [json]
  "Set the xgboost global config from a json string. Set-only side of
  the pair; get-global-config reads back."
  (ok! (XGBSetGlobalConfig json) "set global config")
  nil)

(defn/typed get-global-config {:args [] :ret :string} []
  "The current xgboost global config as a json string. The string is
  owned by the library - valid until the next call, read immediately
  (3.4.1 C API; absent in 2.1.x)."
  (def out (ffi/write :ptr nil))
  (ok! (XGBGetGlobalConfig out) "get global config")
  (ffi/read :string out))

(defn/typed dmatrix-from-file {:args [:string] :ret :struct} [path]
  "A dmatrix from a file uri, e.g. \"<path>?format=libsvm\" (csv also
  supported; the format is the caller's choice - the contract's uri
  convention). Shape keys queried from the loaded matrix. An empty or
  whitespace-only uri raises here: the empty string crashes the C
  library outright (upstream libxgboost defect, reproduced by the
  official python binding)."
  (when (empty? (string/trim path))
    (errorf "xgboost: dmatrix-from-file requires a non-empty path URI"))
  (def out (ffi/write :ptr nil))
  (ok! (XGDMatrixCreateFromURI (json/encode {:uri path}) out) "create dmatrix from uri")
  (def ptr (ffi/read :ptr out))
  (def [nrow ncol] (dmx-shape ptr))
  (make-dmx ptr nrow ncol))

(defn/typed dmx-set-label {:args [:struct :any] :ret :struct} [dmx labels]
  "Attach labels (a sequence of numbers, length nrow) to a dmatrix
  and return it. A length mismatch raises the library's own
  invalid-size message."
  (check-handle dmx :dmatrix)
  (def arr (ffi/write @[:float (length labels)] labels))
  (ok! (XGDMatrixSetFloatInfo (dmx :ptr) "label" arr (length labels)) "set label")
  dmx)

(defn/typed -dmatrix-from-packed* {:args [:struct :any] :ret :struct} [packed labels]
  "Typed core of dmatrix-from-packed: CreateFromMat on a packed f32
  handle (janet-num or duckdb pack-columns-f32), zero janet elements
  crossing; NULLs arrive as NaN = missing. A packed handle without a
  live :ptr raises the type-guard class (a freed pack handle is
  janet-num's lifetime contract, section 7 - not guardable here). The
  :ptr must be a buffer or a true pointer (the
  janet-num pack handle carries a buffer :ptr; anything else is a
  forged struct)."
  (unless (and (or (struct? packed) (table? packed))
              (or (buffer? (get packed :ptr))
                  (= :pointer (type (get packed :ptr)))))
    (errorf "xgboost: expected packed f32 handle, got %s" (type packed)))
  (def out (ffi/write :ptr nil))
  (ok! (XGDMatrixCreateFromMat (packed :ptr) (packed :nrow) (packed :ncol)
                               math/nan out)
       "create dmatrix from packed")
  (def dmx (make-dmx (ffi/read :ptr out) (packed :nrow) (packed :ncol)))
  (if labels (dmx-set-label dmx labels) dmx))

(defn dmatrix-from-packed
  "A dmatrix directly from a packed f32 handle (the duckdb -> xgboost
  C-to-C route; see the contract's dependency section). Optional
  labels attach on creation."
  [packed &opt labels]
  (default labels nil)
  (-dmatrix-from-packed* packed labels))

(defn/typed -dmatrix-from-columns* {:args [:any :any] :ret :struct} [cols labels]
  "Typed core of dmatrix-from-columns: column-major arrays transposed
  to the row-major f32 buffer in one janet-num C pass, then
  CreateFromMat. Guards (empty, ragged) live in the pack kernel."
  (def packed (janet-num/pack-columns-f32 cols))
  (defer (:free packed)
    (def out (ffi/write :ptr nil))
    (ok! (XGDMatrixCreateFromMat (packed :ptr) (packed :nrow) (packed :ncol)
                                 math/nan out)
         "create dmatrix from mat")
    (def dmx (make-dmx (ffi/read :ptr out) (packed :nrow) (packed :ncol)))
    (if labels (dmx-set-label dmx labels) dmx)))

(defn dmatrix-from-columns
  "A dmatrix from column-major arrays (the duckdb columns feed shape).
  Optional labels attach on creation."
  [cols &opt labels]
  (default labels nil)
  (-dmatrix-from-columns* cols labels))

(defn/typed set-param {:args [:struct :any :any] :ret :struct} [bst k v]
  "Set one booster parameter (k and v stringified; names are
  xgboost's own) and return the booster."
  (check-handle bst :booster)
  (ok! (XGBoosterSetParam (bst :ptr) (string k) (string v)) "set param")
  bst)

(defn/typed -booster* {:args [:any :struct] :ret :struct} [dmats params]
  "Typed core of booster: XGBoosterCreate over a list of dmatrix
  handles (empty list is legal - the load-model underlay), then
  params applied entry by entry. Every element must be a dmatrix
  handle (the type-guard class)."
  (each d dmats (check-handle d :dmatrix))
  (def dmx-ptrs (seq [d :in dmats] (d :ptr)))
  (def arr-ptr (ffi/write @[:ptr (length dmx-ptrs)] dmx-ptrs))
  (def out (ffi/write :ptr nil))
  (ok! (XGBoosterCreate arr-ptr (length dmx-ptrs) out) "booster create")
  (def bst (make-bst (ffi/read :ptr out)))
  (eachp [k v] params (set-param bst k v))
  bst)

(defn booster
  "A booster over dmatrix handles (default none) with an optional
  params struct applied at creation."
  [&opt dmats params]
  (default dmats [])
  (default params {})
  (-booster* dmats params))

(defn/typed -train* {:args [:struct :struct :number] :ret :struct} [bst dmx rounds]
  "Typed core of train: UpdateOneIter rounds times, objective-driven
  (the library computes the gradients), iteration counter 0-based."
  (check-handle bst :booster)
  (check-handle dmx :dmatrix)
  (for i 0 rounds
    (ok! (XGBoosterUpdateOneIter (bst :ptr) i (dmx :ptr)) "train iter"))
  bst)

(defn train
  "Train a booster on a labelled dmatrix for rounds iterations
  (default 10; 0 is a legal no-op) and return it."
  [bst dmx &opt rounds]
  (default rounds 10)
  (-train* bst dmx rounds))

(defn/typed predict {:args [:struct :struct] :ret :any} [bst dmx]
  "Predict on a dmatrix with the module's fixed config (value output,
  all trees): 1-dim objectives give a flat array; multi-class gives
  one row array per sample (see the contract's type matrix). A 0-row
  dmatrix returns an empty array (matching the official python
  binding; the library returns rc 0 with null out arrays)."
  (check-handle bst :booster)
  (check-handle dmx :dmatrix)
  (def out-shape (ffi/write :ptr nil))
  (def out-dim (ffi/write :uint64 0))
  (def out-result (ffi/write :ptr nil))
  (ok! (XGBoosterPredictFromDMatrix (bst :ptr) (dmx :ptr) PREDICT-CONFIG
                                    out-shape out-dim out-result)
       "predict")
  (if (or (nil? (ffi/read :ptr out-shape))
          (nil? (ffi/read :ptr out-result)))
    @[]
    (let [dim (int/to-number (ffi/read :uint64 out-dim))
          shape (map int/to-number (ffi/read @[:uint64 dim] (ffi/read :ptr out-shape)))
          n (reduce * 1 shape)
          preds (ffi/read @[:float n] (ffi/read :ptr out-result))]
      (if (= dim 1)
       preds
       (let [cols (shape 1)
              rows @[]]
          (for r 0 (shape 0)
            (array/push rows (slice preds (* r cols) (* (inc r) cols))))
          rows)))))

(defn/typed save-model {:args [:struct :string] :ret :string} [bst path]
  "Save a booster to path (the extension selects the format: .json,
  .ubj) and return path."
  (check-handle bst :booster)
  (ok! (XGBoosterSaveModel (bst :ptr) path) "save model")
  path)

(defn/typed -load-model* {:args [:string :struct] :ret :struct} [path params]
  "Typed core of load-model: fresh empty booster, model loaded from
  path, params applied."
  (def bst (booster))
  (ok! (XGBoosterLoadModel (bst :ptr) path) "load model")
  (eachp [k v] params (set-param bst k v))
  bst)

(defn load-model
  "Load a saved model file into a fresh booster, with an optional
  params struct applied after the load."
  [path &opt params]
  (default params {})
  (-load-model* path params))
