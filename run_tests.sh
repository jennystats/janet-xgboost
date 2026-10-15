#!/bin/sh
# run_tests.sh - janet-xgboost smoke suite. SKIP semantics: absent libs
# skip, present-but-broken is a FAIL, never a skip.
DIR="$(cd "$(dirname "$0")" && pwd)"
[ -n "$JANET_PATH" ] || export JANET_PATH="$HOME/.local/lib/janet"
export JANET_PATH="$DIR:$JANET_PATH"

has_xgboost() { { [ -n "$XGBOOST_LIB" ] || [ -f "$HOME/.local/lib/xgboost/xgboost/lib/libxgboost.so" ] || ldconfig -p 2>/dev/null | grep -q libxgboost; } && { [ -n "$JANET_NUM_LIB" ] || [ -f "$HOME/.local/lib/janet-num/libjn.so" ]; }; }

fail=0
if has_xgboost; then
  if janet "$DIR/test/smoke-xgboost.janet" > /dev/null 2>&1; then
    echo "PASS smoke-xgboost.janet"
  else
    echo "FAIL smoke-xgboost.janet"
    janet "$DIR/test/smoke-xgboost.janet"
    fail=1
  fi
else
  echo "SKIP smoke-xgboost.janet (no libxgboost or libjn - run libs/install-xgboost.sh and the janet-num installer)"
fi

if [ "$fail" -eq 0 ]; then echo "RESULT: pass"; else echo "RESULT: fail"; exit 1; fi
