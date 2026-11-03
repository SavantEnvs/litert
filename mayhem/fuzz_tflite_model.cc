#include <cstddef>
#include <cstdint>
#include <memory>

#include "litert/core/model/verifier.h"
#include "litert/test/fuzzing/fuzzing_util.h"
#include "tflite/core/api/op_resolver.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  litert::LiteRtVerifier verifier;
  tflite::fuzzing::SilentErrorReporter reporter;
  auto model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      reinterpret_cast<const char*>(data), size, &verifier, &reporter);
  if (model) {
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(model->GetModel(), resolver, &reporter)(
        &interpreter);
  }
  return 0;
}
