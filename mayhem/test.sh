#!/usr/bin/env bash
#
# mayhem/test.sh — RUN this repo's OWN functional test suite (already built by mayhem/build.sh).
# exit 0 = pass. EDIT per repo. PATCH-grade oracle: after an agent patches the source, the grader
# rebuilds (build.sh) then runs this. DELETE this file if the repo has no meaningful tests.
#
# IMPORTANT:
#  * Must assert BEHAVIOR/OUTPUT, not just exit status. The oracle has to check asserted values /
#    golden-output diffs / known-answer results — so a PATCH that "fixes" a bug by making the program
#    exit(0) (or any no-op) FAILS here. Running inputs and checking only "exit 0 / didn't crash" is
#    NOT a functional test (it's trivially reward-hackable) — use the project's real assertion suite.
#  * Do NOT build here — mayhem/build.sh already compiled the test suite (with the project's normal
#    flags). This script only RUNS the pre-built tests and reports counts. If the test runner is
#    missing, that's a build.sh bug — fail loudly rather than silently rebuilding.
#  * REQUIRED OUTPUT — a CTRF (https://ctrf.io) summary so Mayhem/the PATCH grader reads the counts:
#      - writes a CTRF JSON report to ${CTRF_REPORT:-$SRC/ctrf-report.json}, and
#      - prints a one-line `CTRF {...}` marker to stdout (same JSON, compact).
#    Only `results.summary` (with tests/passed/failed/pending/skipped/other) is required.
#    Use the emit_ctrf helper below; it computes tests = passed+failed+skipped and sets the exit
#    code (0 iff failed==0). Map your framework's output to passed/failed/skipped.
set -uo pipefail
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH
: "${MAYHEM_JOBS:=$(nproc)}"   # build parallelism; env-overridable, falls back to nproc (use -j"$MAYHEM_JOBS")
cd "$SRC"

# emit_ctrf <tool> <passed> <failed> [skipped] [pending] [other]
# Writes a CTRF report (file + stdout `CTRF {...}` marker) and returns non-zero iff failed>0.
emit_ctrf() {
  local tool="$1" passed="$2" failed="$3" skipped="${4:-0}" pending="${5:-0}" other="${6:-0}"
  local tests=$(( passed + failed + skipped + pending + other ))
  cat > "${CTRF_REPORT:-$SRC/ctrf-report.json}" <<JSON
{
  "results": {
    "tool": { "name": "$tool" },
    "summary": {
      "tests": $tests,
      "passed": $passed,
      "failed": $failed,
      "pending": $pending,
      "skipped": $skipped,
      "other": $other
    }
  }
}
JSON
  printf 'CTRF {"results":{"tool":{"name":"%s"},"summary":{"tests":%d,"passed":%d,"failed":%d,"pending":%d,"skipped":%d,"other":%d}}}\n' \
    "$tool" "$tests" "$passed" "$failed" "$pending" "$skipped" "$other"
  [ "$failed" -eq 0 ]
}

ORACLE="${MAYHEM_OUT:-$SRC}/litert_oracle_test"
CTEST_DIR="${LITERT_CACHE:-/opt/toolchains/litert}/build-norm/tflite_build"
if [ ! -x "$ORACLE" ] || [ ! -f "$CTEST_DIR/kernels/CTestTestfile.cmake" ]; then
  echo "missing $ORACLE or $CTEST_DIR/kernels/CTestTestfile.cmake (mayhem/build.sh should have built them)" >&2
  emit_ctrf "litert-oracle+gtest" 0 1 0
  exit 1
fi

# 1) Known-answer oracle over LiteRT's model loader/verifier (mayhem/oracle_test.cc).
log="$(mktemp)"
"$ORACLE" > "$log" 2>&1
rc=$?
cat "$log"
passed=$(grep -c '^PASS: ' "$log")
failed=$(grep -c '^FAIL: ' "$log")
rm -f "$log"
if [ "$rc" -ne 0 ] && [ "$failed" -eq 0 ]; then failed=1; fi
if [ "$passed" -eq 0 ]; then failed=$((failed + 1)); fi

# 2) Upstream's own TFLite kernel tests (gtest, built by build.sh via TFLITE_KERNEL_TEST), through ctest.
log="$(mktemp)"
ctest --test-dir "$CTEST_DIR" -V -R '^(cast_test|add_test)$' > "$log" 2>&1
rc=$?
cat "$log"
gpassed=$(grep -E '^[0-9]+: \[  PASSED  \] [0-9]+ test' "$log" | awk '{s+=$5} END {print s+0}')
gfailed=$(grep -E '^[0-9]+: \[  FAILED  \] [0-9]+ test' "$log" | awk '{s+=$5} END {print s+0}')
gskipped=$(grep -E '^[0-9]+: \[  SKIPPED \] [0-9]+ test' "$log" | awk '{s+=$5} END {print s+0}')
rm -f "$log"
if [ "$rc" -ne 0 ] && [ "$gfailed" -eq 0 ]; then gfailed=1; fi
if [ "$gpassed" -eq 0 ]; then gfailed=$((gfailed + 1)); fi

emit_ctrf "litert-oracle+gtest" "$((passed + gpassed))" "$((failed + gfailed))" "$gskipped"
