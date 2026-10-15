# smoke-xgboost - conformance: the full public surface at the duckdb
# depth standard (lifecycle, both ingest routes, error taxonomy,
# global config round trip, recorded exact predictions, save/load).
# Requires libxgboost (SKIP-if-absent semantics live in
# run_tests.sh; this file runs only when the lib exists).
# Behavioural spec: specs/api-contract.md v1.3.1.
#
# Expected values recorded from the pinned official wheel lib 2.1.4
# and verified unchanged on 3.4.1 (nthread=1, seed=42 -> bitwise-stable
# across runs and processes).
(import janet-xgboost :as xgboost)
(import typed :prefix "")
(import janet-num)

(defn assert-eq [a b msg]
  (unless (deep= a b) (errorf "FAIL %s: %q != %q" msg a b)))

(defn assert-raises [f msg-match]
  (var caught nil)
  (try (f) ([e] (set caught e)))
  (unless (and caught (string/find msg-match caught))
    (errorf "FAIL expected error matching %q, got %q" msg-match caught)))

# -- permanent fixtures ---------------------------------------------------------
# The libsvm fixture and the recorded expectations are named here and
# reused by every cell; nothing inline-scattered (the duckdb standard).
(def FEATURES-COLS [[0.1 0.9 0.4 0.7] [0.2 0.8 0.5 0.1]])
(def LABELS [1.0 4.0 2.0 3.0])
(def PARAMS {:objective "reg:squarederror" :max_depth "2"
             :eta "0.3" :seed "42" :nthread "1"})
(def RECORDED-PREDS
  @[1.3011019229888916 3.6988980770111084
    2.0694849491119385 2.9305150508880615])
(def LIBSVM-TEXT "1 0:0.1 1:0.2\n4 0:0.9 1:0.8\n2 0:0.4 1:0.5\n3 0:0.7 1:0.1\n")
(def LIBSVM-PATH "/tmp/xgboost-smoke.libsvm")
(def MODEL-PATH "/tmp/xgboost-smoke-model.json")

# -- version pin (install coupling) ----------------------------------------------
(assert-eq (xgboost/version) [3 4 1] "pinned version 3.4.1")

# -- global config: fresh-process default, then round trip -----------------------
# This cell is only valid because the smoke process has not yet set
# any global config - it must stay above the first global-config call.
# Observed default on 3.4.1: a populated json with verbosity 1, not {}.
(assert (string/find "\"verbosity\":1" (xgboost/get-global-config))
        "get-global-config default before any set (fresh process)")
(assert-eq (xgboost/global-config "{\"verbosity\":0}") nil
           "global-config set returns nil")
(assert (string/find "\"verbosity\":0" (xgboost/get-global-config))
        "global-config set then get round trip")
(assert-raises (fn [] (xgboost/global-config "{not json"))
               "set global config failed")
(assert-raises (fn [] (xgboost/global-config "{\"verbosity\":\"abc\"}"))
               "Invalid Parameter format for verbosity")

# -- columnar route: dmatrix-from-columns -----------------------------------------
(def dmx (xgboost/dmatrix-from-columns FEATURES-COLS LABELS))
(assert-eq [(dmx :nrow) (dmx :ncol)] [4 2] "dmatrix shape keys")
(assert (not (nil? (dmx :ptr))) "dmatrix carries a raw pointer")
(assert (function? (dmx :close)) "dmatrix carries idempotent :close")

# -- packed route: dmatrix-from-packed ---------------------------------------------
# A janet-num pack handle crosses with zero janet elements; the packed
# route must be bitwise-identical to the columns route on the same
# data (the contract's parity guarantee, section 7).
(def packed (janet-num/pack-columns-f32 FEATURES-COLS))
(assert-eq [(packed :nrow) (packed :ncol)] [4 2] "pack handle shape")
(def pdmx (xgboost/dmatrix-from-packed packed LABELS))
(assert-eq [(pdmx :nrow) (pdmx :ncol)] [4 2] "packed dmatrix shape keys")
(assert (not (nil? (pdmx :ptr))) "packed dmatrix carries a raw pointer")

# -- dmx-set-label standalone -------------------------------------------------------
# The label route at create and via dmx-set-label is the same C path
# (contract section 4); the standalone form is pinned by parity with
# the create-time labels through a full train.
(def dmx-nolabel (xgboost/dmatrix-from-columns FEATURES-COLS))
(assert-eq (xgboost/dmx-set-label dmx-nolabel LABELS) dmx-nolabel
           "dmx-set-label chains (returns the handle)")

# -- booster forms -------------------------------------------------------------------
# The empty-dmats form is the load-model underlay (contract section 2):
# legal with no matrices and no params, params applied via set-param.
(def ebst (xgboost/booster))
(assert (not (nil? (ebst :ptr))) "empty booster form carries a pointer")
(assert (function? (ebst :close)) "empty booster carries idempotent :close")
(assert-eq (xgboost/set-param ebst "seed" "42") ebst
           "set-param chains (returns the handle)")
# train chains too; 0 rounds is a legal no-op (contract section 2).
(def bst0 (xgboost/train (xgboost/booster [dmx] PARAMS) dmx 0))
(assert-eq (xgboost/predict bst0 dmx)
           @[0.5 0.5 0.5 0.5]
           "0 rounds trains nothing (untrained base margin, observed 0.5)")

# -- deterministic train + recorded predictions --------------------------------------
(def bst (xgboost/train (xgboost/booster [dmx] PARAMS) dmx 10))
(assert-eq (xgboost/predict bst dmx) RECORDED-PREDS
           "recorded predictions (wheel lib, seed 42, 10 rounds)")

# train defaults to 10 rounds when omitted (contract section 2)
(def bst-def (xgboost/train (xgboost/booster [dmx] PARAMS) dmx))
(assert-eq (xgboost/predict bst-def dmx) RECORDED-PREDS
           "train default rounds is 10")

# the standalone-label dmatrix trains to the same predictions
(def bst-sl (xgboost/train (xgboost/booster [dmx-nolabel] PARAMS)
                           dmx-nolabel 10))
(assert-eq (xgboost/predict bst-sl dmx-nolabel) RECORDED-PREDS
           "dmx-set-label route matches create-time labels")

# the packed route is bitwise-identical end to end
(def bstp (xgboost/train (xgboost/booster [pdmx] PARAMS) pdmx 10))
(assert-eq (xgboost/predict bstp pdmx) RECORDED-PREDS
           "packed route parity with recorded values")

# retrain from scratch in the same process -> bitwise identical
(def bst2 (xgboost/train (xgboost/booster [dmx] PARAMS) dmx 10))
(assert-eq (xgboost/predict bst2 dmx) RECORDED-PREDS "retrain deterministic")

# -- set-param behaviour on 3.4.1 -----------------------------------------------------
# Observed on 3.4.1 via XGBoosterSetParam: an unknown key is accepted
# silently (validation happens at use, not at set) - the contract's
# params row documents this behaviour.
(def dev-bst (xgboost/set-param (xgboost/booster) "not-a-real-param" "1"))
(assert (not (nil? (dev-bst :ptr)))
        "unknown param key is accepted silently on 3.4.1 (documented in the contract's params row)")
# A malformed VALUE surfaces at train time, with the library's own
# diagnostic through the module's one error class.
(assert-raises
  (fn []
    (xgboost/train (xgboost/booster [dmx] {"eta" "abc"}) dmx 1))
  "Invalid Parameter format for learning_rate")

# -- save / load round trip -----------------------------------------------------------
(def saved (xgboost/save-model bst MODEL-PATH))
(assert-eq saved MODEL-PATH "save returns path")
(def loaded (xgboost/load-model MODEL-PATH PARAMS))
(assert-eq (xgboost/predict loaded dmx) RECORDED-PREDS
           "loaded model predicts identically")

# -- file route (libsvm uri) ------------------------------------------------------------
(spit LIBSVM-PATH LIBSVM-TEXT)
(def fdmx (xgboost/dmatrix-from-file (string LIBSVM-PATH "?format=libsvm")))
(assert-eq [(fdmx :nrow) (fdmx :ncol)] [4 2] "from-file shape keys")
(assert-eq (xgboost/predict loaded fdmx) RECORDED-PREDS
           "libsvm file dmatrix predicts identically")

# -- missing values (NaN cells) ----------------------------------------------------------
# xgboost's native missing semantics: the module's create missing
# value is NaN (contract section 4), so a NaN feature cell is accepted
# without error and predictions stay finite. Pinned on both ingest
# routes; the packed route with a NaN cell still matches recorded
# values for the other rows' structure.
(def NAN-COLS [[0.1 0.9 0.4 0.7] [math/nan 0.8 0.5 0.1]])
(def nan-dmx (xgboost/dmatrix-from-columns NAN-COLS LABELS))
(def nan-preds (xgboost/predict
                 (xgboost/train (xgboost/booster [nan-dmx] PARAMS)
                                nan-dmx 10)
                 nan-dmx))
(assert-eq (length nan-preds) 4 "NaN cells: prediction count intact")
(each p nan-preds (assert (not (= p math/nan)) "NaN cells: predictions finite"))
(def packed-nan (janet-num/pack-columns-f32 NAN-COLS))
(def nan-pdmx (xgboost/dmatrix-from-packed packed-nan LABELS))
(def nan-ppreds (xgboost/predict
                  (xgboost/train (xgboost/booster [nan-pdmx] PARAMS)
                                 nan-pdmx 10)
                  nan-pdmx))
(assert-eq nan-ppreds nan-preds "NaN cells: packed and columns routes agree")

# -- 0-row predict (official python binding behaviour) ----------------
# Constructors accept 0-row input (shape [0 n]); the library returns
# rc 0 with null out arrays on predict and the wrapper returns an
# empty array (matching the official python binding; a stderr
# warning is the library's own and expected).
(def z-dmx (xgboost/dmatrix-from-columns [[] []]))
(assert-eq [(z-dmx :nrow) (z-dmx :ncol)] [0 2] "0-row dmatrix constructs")
(assert-eq (xgboost/predict bst z-dmx) @[] "0-row predict returns empty array")
(def z-pdmx (xgboost/dmatrix-from-packed (janet-num/pack-columns-f32 [[] []])))
(assert-eq (xgboost/predict bst z-pdmx) @[]
           "0-row packed-route predict returns empty array")

# -- multi-output shape (2-dim predict -> rows) ------------------------------------------
(def mdmx (xgboost/dmatrix-from-columns [[0.1 0.9 0.4 0.7]] [0 1 0 1]))
(def mbst (xgboost/train (xgboost/booster [mdmx]
                           {:objective "multi:softprob" :num_class "2"
                            :max_depth "1" :eta "0.3" :seed "42"
                            :nthread "1"})
                         mdmx 2))
(def mpreds (xgboost/predict mbst mdmx))
(assert (= 4 (length mpreds)) "2-dim predict gives one row per sample")
(assert (= 2 (length (mpreds 0))) "each row has num_class entries")
(each r mpreds (assert (< 0.99 (sum r) 1.01) "softprob rows sum to 1"))

# -- `with` sugar (both handle kinds close through :close) ---------------------------------
(assert-eq (with [wdmx (xgboost/dmatrix-from-columns FEATURES-COLS)]
             [(wdmx :nrow) (wdmx :ncol)])
           [4 2]
           "with closes a dmatrix handle after the body")
(assert-eq (with [wbst (xgboost/booster)] :body-value)
           :body-value
           "with closes a booster handle after the body")

# -- error taxonomy (contract section 5) ----------------------------------------------------
# XG class: one fixed <what> per call site, library diagnostic carried.
(assert-raises (fn [] (xgboost/load-model "/nonexistent-model.json")) "failed")
(assert-raises (fn [] (xgboost/dmatrix-from-file "/nonexistent.libsvm")) "failed")
(assert-raises (fn [] (xgboost/dmatrix-from-columns [])) "no columns")
(assert-raises (fn [] (xgboost/dmatrix-from-columns [[1 2] [1 2 3]])) "ragged")
# label length mismatch raises in the C layer's set-info call, both
# at create and via dmx-set-label (the same C path).
(assert-raises (fn [] (xgboost/dmatrix-from-columns FEATURES-COLS
                                                    [1.0 2.0 3.0]))
               "set label failed")
(assert-raises (fn [] (xgboost/dmx-set-label dmx [1.0 2.0 3.0]))
               "set label failed")
# VALIDATION (native): a non-number element raises in janet's
# ffi/write with the offending element index (contract section 5).
(assert-raises (fn [] (xgboost/dmatrix-from-columns [["a" "b"]] nil))
               "bad slot #0, expected number")
# TYPE-GUARD class (contract section 5, added v1.2): every
# dmatrix/booster-taking fn raises on a wrong-kind or missing-:ptr
# argument. The crash cells were observed subprocess-red (exit 139)
# on the pre-guard tree; the guard converts them to this class - no
# exit 139 anywhere in the surface (wrong-kind and closed-handle
# use-after-close are distinct: closed handles still carry :kind,
# so close-then-discard UB is unchanged, contract section 3).
(assert-raises (fn [] (xgboost/booster @[@{:nrow 2 :ncol 2}]))
               "expected dmatrix handle, got table")
(assert-raises (fn [] (xgboost/booster [nil]))
               "expected dmatrix handle, got nil")
(assert-raises (fn [] (xgboost/predict dmx bst))
               "expected booster handle, got struct")
(assert-raises (fn [] (xgboost/train (xgboost/booster) {:nrow 2}))
               "expected dmatrix handle, got struct")
(assert-raises (fn [] (xgboost/set-param dmx "seed" "1"))
               "expected booster handle, got struct")
(assert-raises (fn [] (xgboost/dmx-set-label bst [1.0]))
               "expected dmatrix handle, got struct")
(assert-raises (fn [] (xgboost/save-model {:ptr 1} "/tmp/x"))
               "expected booster handle, got struct")
(assert-raises (fn [] (xgboost/dmatrix-from-packed {:nrow 2 :ncol 2}))
               "expected packed f32 handle, got struct")
# FORGED-STRUCT class (closed in v1.3): a struct with
# the right :kind but a non-pointer :ptr passes presence-only guards
# and reaches the C layer - subprocess-red pre-fix: predict/train/
# label/set-param/save-model over {:kind :dmatrix :ptr :k ...} exit
# 139; booster-over-forged-element and dmatrix-from-packed over a
# non-buffer non-pointer :ptr exit 0 with a garbage handle (silent
# corruption). The pointer-ness requirement converts both classes to
# the type-guard class; same message shape, taxonomy unchanged.
(assert-raises (fn [] (xgboost/predict bst {:kind :dmatrix :ptr :k :nrow 2 :ncol 2}))
               "expected dmatrix handle, got struct")
(assert-raises (fn [] (xgboost/train bst {:kind :dmatrix :ptr :k :nrow 2 :ncol 2}))
               "expected dmatrix handle, got struct")
(assert-raises (fn [] (xgboost/dmx-set-label {:kind :dmatrix :ptr :k :nrow 2 :ncol 2} [1.0]))
               "expected dmatrix handle, got struct")
(assert-raises (fn [] (xgboost/set-param {:kind :booster :ptr :k} "seed" "1"))
               "expected booster handle, got struct")
(assert-raises (fn [] (xgboost/save-model {:kind :booster :ptr :k} "/tmp/x"))
               "expected booster handle, got struct")
# silent-corruption class: must RAISE, never exit 0 with a handle
(assert-raises (fn [] (xgboost/booster [{:kind :dmatrix :ptr :k :nrow 1 :ncol 1}]))
               "expected dmatrix handle, got struct")
(assert-raises (fn [] (xgboost/dmatrix-from-packed {:kind :packed :ptr :k :nrow 2 :ncol 2}))
               "expected packed f32 handle, got struct")
(assert-raises (fn [] (xgboost/dmatrix-from-packed {:kind :packed :ptr "x" :nrow 2 :ncol 2}))
               "expected packed f32 handle, got struct")
# (a buffer :ptr is legal here - janet-num pack handles carry one;
# short-buffer sufficiency is janet-num's own guard class, not this
# module's)
(assert-raises (fn [] (xgboost/predict {:kind :booster :ptr (fn [] 1)} dmx))
               "expected booster handle, got struct")
# Empty-URI guard (added v1.2): the empty string segfaults the C
# library outright (upstream defect, reproduced by the official python
# binding; subprocess-red pre-fix); the wrapper rejects it before the
# ffi call, whitespace-only included.
(assert-raises (fn [] (xgboost/dmatrix-from-file ""))
               "requires a non-empty path URI")
(assert-raises (fn [] (xgboost/dmatrix-from-file " "))
               "requires a non-empty path URI")
# ENVIRONMENT class (lib absent) is not raisable in-suite: the lib is
# present by the time this file runs. Marked not-raisable here; the
# message shape is pinned by the module's load guard instead.
#
# Use-after-close is deliberately NOT a cell: the contract (section 3)
# documents no guard - a closed handle's pointer is freed and reuse is
# undefined behaviour, observed to segfault the process (probe,
# 2026-10-16). The user contract is close-then-discard. The lifecycle
# cells below pin everything that IS safe: double-close idempotence
# and the with sugar.

# -- lifecycle: double-close idempotence on every handle class -------------
(:close bst) (:close bst)
(:close bst2) (:close bst2)
(:close bst0) (:close bst0)
(:close bst-def) (:close bst-def)
(:close bst-sl) (:close bst-sl)
(:close bstp) (:close bstp)
(:close ebst) (:close ebst)
(:close dev-bst) (:close dev-bst)
(:close loaded) (:close loaded)
(:close dmx) (:close dmx)
(:close dmx-nolabel) (:close dmx-nolabel)
(:close pdmx) (:close pdmx)
(:close fdmx) (:close fdmx)
(:close nan-dmx) (:close nan-dmx)
(:close nan-pdmx) (:close nan-pdmx)
(:close mdmx) (:close mdmx)
(:close mbst) (:close mbst)
(:close z-dmx) (:close z-dmx)
(:close z-pdmx) (:close z-pdmx)
(:free packed) (:free packed)
(:free packed-nan) (:free packed-nan)
(print "xgboost smoke ok")
