#!/usr/bin/env bash
#
# mayhem/build.sh — build this repo's fuzz harness(es). EDIT per repo.
#
# Runs inside the commit image (mayhem/Dockerfile) as `mayhem` in /mayhem. The base image
# (ghcr.io/savantenvs/base) already exports the build contract — use these, don't redefine:
#   CC, CXX             stock clang / clang++
#   LIB_FUZZING_ENGINE  -fsanitize=fuzzer   (link into each harness that has a LLVMFuzzer entry)
#   SANITIZER_FLAGS     -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
#                       (ASan + UBSan, both set to HALT — so Mayhem catches memory AND UB defects)
#   DEBUG_FLAGS         -g -gdwarf-3   (DWARF debug info — always on for fuzz/standalone builds,
#                       independent of the sanitizer off-switch; DWARF version must be < 4)
#   RUST_DEBUG_FLAGS    -C debuginfo=2 -C force-frame-pointers=yes -C llvm-args=-gdwarf-3
#                       (thread through RUSTFLAGS on every cargo-fuzz build)
#   GO_DEBUG_FLAGS      -gcflags=all=-N -l
#                       (thread through go build / go-fuzz-build so the linked ELF keeps symbols)
#   SRC                 /mayhem (the repo source)
#
# Contract: build EVERYTHING here — one runnable binary per fuzz harness, AND the project's test
# suite (so mayhem/test.sh only has to RUN it, never compile). Keep it ADDITIVE (build upstream as
# upstream documents; don't edit upstream files). IMPORTANT: build the PROJECT ITSELF with
# $SANITIZER_FLAGS and $DEBUG_FLAGS (not just the harness) so the fuzzed code is instrumented
# AND carries DWARF < 4 symbols — otherwise ASan/UBSan only see the harness, not the library
# you're trying to find bugs in, and backtraces won't resolve project source lines. Build the TEST
# suite with the project's NORMAL flags (a clean, independent build) so test.sh stays an honest
# functional oracle and won't false-fail on benign UB. Leave the test binary/runner where test.sh
# expects it.
set -euo pipefail

# clang rejects SOURCE_DATE_EPOCH='' (empty) — it must be unset or a valid integer.
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH

# Build knobs come from the ENVIRONMENT (overridable), with sane defaults — no if-statements,
# just parameter-expansion fallbacks. Default sanitizers are the base's ASan+UBSan (halting); override
# per build via the Dockerfile's `--build-arg SANITIZER_FLAGS="..."`.
# NB: SANITIZER_FLAGS uses `=` (no colon) on purpose — `=` only fills when the var is UNSET, so an
# explicit EMPTY value (`--build-arg SANITIZER_FLAGS=`) is honored and builds with NO sanitizers
# (useful when you want the program's natural crash / full backtrace, not an ASan report). The other
# knobs use `:=` (default on empty too). MAYHEM_JOBS sets build parallelism (falls back to nproc).
: "${SANITIZER_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer}"
# DEBUG_FLAGS carries DWARF debug info INDEPENDENTLY of the sanitizer off-switch (so an empty
# SANITIZER_FLAGS still yields DWARF symbols). DWARF MUST be < 4 (Mayhem triage can't read >=4); clang-19's
# plain `-g` emits DWARF-5, so `-gdwarf-3` is explicit. Apply $DEBUG_FLAGS to the fuzz/harness/standalone
# builds (NOT the test/oracle build). Rust/Go carry the same intent via their language flags below.
: "${DEBUG_FLAGS:=-g -gdwarf-3}"
: "${RUST_DEBUG_FLAGS:=-C debuginfo=2 -C force-frame-pointers=yes -C llvm-args=-gdwarf-3}"
: "${GO_DEBUG_FLAGS:=-gcflags=all=-N -l}"
: "${CC:=clang}" ; : "${CXX:=clang++}" ; : "${LIB_FUZZING_ENGINE:=-fsanitize=fuzzer}"
: "${MAYHEM_JOBS:=$(nproc)}"
# COVERAGE_FLAGS: empty by default → no effect on the normal oracle build. Set it via the Dockerfile's
# `--build-arg COVERAGE_FLAGS="-fprofile-instr-generate -fcoverage-mapping"` to instrument the TEST
# build for source-coverage measurement (how much of the project the test suite actually exercises —
# a quality signal for the oracle; complements the anti-reward-hack sabotage check). APPEND it to the
# test build's compile+link flags in step 3 (NOT the fuzz build); after `test.sh` runs, merge with
# `llvm-profdata` and report with `llvm-cov`. Empty value is honored (`=`, not `:=`).
: "${COVERAGE_FLAGS=}"
export SANITIZER_FLAGS DEBUG_FLAGS RUST_DEBUG_FLAGS GO_DEBUG_FLAGS CC CXX LIB_FUZZING_ENGINE MAYHEM_JOBS COVERAGE_FLAGS

cd "$SRC"

CACHE="${LITERT_CACHE:-/opt/toolchains/litert}"
SAN_DIR="$CACHE/build-san"
NORM_DIR="$CACHE/build-norm"
OUT="${MAYHEM_OUT:-$SRC}"
mkdir -p "$CACHE"

# The CMake build FetchContent-downloads ~20 pinned deps on the first (online) configure. They live inside
# the build dirs under /opt/toolchains (outside the git tree, so `git clean -ffdX` cannot remove them);
# once configured, the dirs are flipped to FETCHCONTENT_FULLY_DISCONNECTED so every later run is offline.
# tensor/examples/gemma3 (configured by litert/CMakeLists.txt, never built here) FetchContent-downloads sentencepiece;
# use the copy the Dockerfile pre-fetched. It needs a writable copy because that CMake patches the sources in place.
SPM_SRC="$CACHE/sentencepiece-src"
[ -d "$SPM_SRC" ] || { cp -r /opt/toolchains/deps/sentencepiece-0.2.0/src "$SPM_SRC" && chmod -R u+w "$SPM_SRC"; }
COMMON_CMAKE=(
  -S litert -G Ninja -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
  -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_GPU=OFF
  -DTFLITE_ENABLE_NNAPI=OFF -DLITERT_ENABLE_GPU=OFF -DLITERT_ENABLE_NPU=OFF -Wno-dev
  -DFETCHCONTENT_SOURCE_DIR_LITERT_SENTENCEPIECE="$SPM_SRC"
)
configure() {
  local dir="$1" btype="$2" flags="$3"; shift 3
  if [ ! -f "$dir/.mayhem-configured" ]; then
    rm -rf "$dir"
    cmake "${COMMON_CMAKE[@]}" -B "$dir" "-DCMAKE_BUILD_TYPE=$btype" "-DCMAKE_C_FLAGS=$flags" "-DCMAKE_CXX_FLAGS=$flags" "$@"
    cmake -DFETCHCONTENT_FULLY_DISCONNECTED=ON "$dir" >/dev/null
    touch "$dir/.mayhem-configured"
  fi
}

# 1) Project (tflite runtime + LiteRT core model) with sanitizers + DWARF 3.
# Coverage instrumentation for the library itself (the libFuzzer engine is linked only into the harnesses).
# nonnull-attribute is relaxed: LiteRtModelT::PushMetadata copies empty metadata via memcpy(dst, nullptr, 0)
# (litert_buffer_ref.h), which halts every model that carries an empty metadata buffer before deeper code runs.
COV_FLAGS=""
[ -z "$SANITIZER_FLAGS" ] || COV_FLAGS="-fsanitize=fuzzer-no-link -fno-sanitize=nonnull-attribute"
# Build type None, not Release: Release adds -O3 -DNDEBUG, which compiles out the library's ABSL_DCHECKs and
# TFLITE_ASSERT_FALSE and overrides the -O1 here. With None, only these flags apply and the asserts stay live.
configure "$SAN_DIR" None "$SANITIZER_FLAGS $COV_FLAGS $DEBUG_FLAGS -O1"
# With the asserts live, the library's CHECK_EQ/ABSL_DCHECK_* reference abseil's check-op and log message code,
# which nothing else pulls in (it is dead code under NDEBUG), so build abseil's log libraries explicitly.
mapfile -t ABSL_LOG < <(ninja -C "$SAN_DIR" -t targets all | sed -n 's/^\(_deps\/abseil-cpp-build\/absl\/log\/libabsl_[a-z0-9_]*\.a\):.*/\1/p' | grep -v -e log_flags -e log_initialize)
[ "${#ABSL_LOG[@]}" -gt 0 ] || { echo "no abseil log targets found" >&2; exit 1; }
ninja -C "$SAN_DIR" -j"$MAYHEM_JOBS" tensorflow-lite litert_core_model "${ABSL_LOG[@]}"

# Compile flags/includes exactly as CMake used them for litert's own model sources.
extract_flags() {
  local all cmd tok skip=0
  all="$(ninja -C "$1" -t commands litert_core_model)"
  cmd="$(grep -m1 'model_load\.cc$' <<<"$all")"
  FLAGS=()
  # shellcheck disable=SC2206
  local toks=($cmd)
  for tok in "${toks[@]:1}"; do
    if [ "$skip" -eq 1 ]; then skip=0; continue; fi
    case "$tok" in -MD) ;; -MT|-MF|-o) skip=1 ;; -c) break ;; *) FLAGS+=("$tok") ;; esac
  done
}

# 2) Harnesses (fuzz + standalone), linked against the sanitized libraries.
extract_flags "$SAN_DIR"
WORK="$SAN_DIR/mayhem-harness"
mkdir -p "$WORK"
cc_obj() { "$CXX" "${FLAGS[@]}" "$@"; }
cc_obj -c mayhem/lsan_off.cc -o "$WORK/lsan_off.o"
$CC $SANITIZER_FLAGS $DEBUG_FLAGS -c "$STANDALONE_FUZZ_MAIN" -o "$WORK/standalone_main.o"
for src in litert/core/model/verifier.cc litert/test/fuzzing/one_op_fuzz_model.cc litert/test/fuzzing/fuzzer_quota_allocator.cc; do
  cc_obj -c "$src" -o "$WORK/$(basename "${src%.cc}").o"
done

mapfile -t ARCHIVES < <(find "$SAN_DIR" -name '*.a' | sort)
link() {  # link <name> <extra-objs...>
  local name="$1"; shift
  cc_obj -c "mayhem/$name.cc" -o "$WORK/$name.o"
  "$CXX" "${FLAGS[@]}" "$WORK/$name.o" "$@" "$WORK/lsan_off.o" \
    -Wl,--start-group "${ARCHIVES[@]}" -Wl,--end-group -lpthread -ldl -lm \
    -o "$WORK/$name.bin"
}
for name in fuzz_tflite_model fuzz_litert_model fuzz_tflite_cast; do
  case "$name" in
    fuzz_tflite_cast) EXTRA=("$WORK/verifier.o" "$WORK/one_op_fuzz_model.o" "$WORK/fuzzer_quota_allocator.o") ;;
    *) EXTRA=("$WORK/verifier.o") ;;
  esac
  link "$name" $LIB_FUZZING_ENGINE "${EXTRA[@]}"
  cp "$WORK/$name.bin" "$OUT/$name"
  link "$name" "$WORK/standalone_main.o" "${EXTRA[@]}"
  cp "$WORK/$name.bin" "$OUT/$name-standalone"
done

# 3) Test suite: known-answer oracle over the same sources, built with NORMAL flags (no sanitizers) in
#    a separate build tree. test.sh only runs $OUT/litert_oracle_test.
#    The same tree also builds a subset of upstream's own CMake kernel tests (TFLITE_KERNEL_TEST); test.sh runs them
#    through ctest from $NORM_DIR/tflite_build. The other upstream tests are Bazel-only (litert/core/model has no
#    CMake test targets), so they cannot be built here.
UPSTREAM_TESTS=(cast_test add_test)
# TFLITE_ENABLE_XNNPACK=ON: upstream's CMake test framework compiles tools/delegates/xnnpack_delegate_provider.cc
# unconditionally, and that file does not compile with XNNPACK off.
configure "$NORM_DIR" Release "$COVERAGE_FLAGS" -DTFLITE_KERNEL_TEST=ON -DTFLITE_ENABLE_XNNPACK=ON
ninja -C "$NORM_DIR" -j"$MAYHEM_JOBS" tensorflow-lite litert_core_model "${UPSTREAM_TESTS[@]}"
extract_flags "$NORM_DIR"
NWORK="$NORM_DIR/mayhem-oracle"
mkdir -p "$NWORK"
for src in mayhem/oracle_test.cc litert/core/model/verifier.cc; do
  cc_obj -c "$src" -o "$NWORK/$(basename "${src%.cc}").o"
done
mapfile -t NARCHIVES < <(find "$NORM_DIR" -name '*.a' | sort)
"$CXX" "${FLAGS[@]}" "$NWORK/oracle_test.o" "$NWORK/verifier.o" \
  -Wl,--start-group "${NARCHIVES[@]}" -Wl,--end-group -lpthread -ldl -lm \
  -o "$OUT/litert_oracle_test"
