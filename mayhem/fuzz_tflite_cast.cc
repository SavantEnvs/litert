#include <fuzzer/FuzzedDataProvider.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include "litert/test/fuzzing/fuzzing_util.h"
#include "litert/test/fuzzing/one_op_fuzz_model.h"
#include "tflite/c/common.h"
#include "tflite/kernels/builtin_op_kernels.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace builtin {
TfLiteRegistration* Register_CAST();
}  // namespace builtin
}  // namespace ops
namespace {

constexpr size_t kMaxInputElements = 512;
constexpr size_t kMaxLiveAllocationBytes = 64 * 1024 * 1024;

constexpr TensorType kTypes[] = {
    TensorType_FLOAT32,  TensorType_FLOAT16, TensorType_INT32,
    TensorType_UINT8,    TensorType_INT64,   TensorType_BOOL,
    TensorType_INT16,    TensorType_COMPLEX64, TensorType_INT8,
    TensorType_FLOAT64,  TensorType_UINT32,  TensorType_UINT16,
    TensorType_INT4,     TensorType_BFLOAT16, TensorType_INT2,
    TensorType_UINT4};

bool IsPacked(TensorType t) {
  return t == TensorType_INT4 || t == TensorType_INT2 || t == TensorType_UINT4;
}
bool IsFloating(TensorType t) {
  return t == TensorType_FLOAT32 || t == TensorType_FLOAT16 ||
         t == TensorType_FLOAT64;
}

void RunCast(FuzzedDataProvider& fdp) {
  TensorType input_type = fdp.PickValueInArray(kTypes);
  TensorType output_type = fdp.PickValueInArray(kTypes);
  if (IsPacked(input_type)) output_type = TensorType_FLOAT32;
  if (IsPacked(output_type)) input_type = TensorType_FLOAT32;

  std::vector<int32_t> shape;
  const size_t rank = fdp.ConsumeIntegralInRange<size_t>(0, 5);
  for (size_t i = 0; i < rank; ++i) {
    shape.push_back(fdp.ConsumeIntegralInRange<int32_t>(-1, 3));
  }
  std::vector<uint8_t> input_data = fdp.ConsumeBytes<uint8_t>(64);

  if (IsFloating(input_type) && !IsFloating(output_type) &&
      output_type != TensorType_BOOL) {
    input_data.clear();
  }
  if (fuzzing::TypeSize(input_type) == 0 ||
      fuzzing::TypeSize(output_type) == 0) {
    return;
  }
  size_t elements = 0;
  if (!fuzzing::CheckedShapeElementCount(shape, &elements) ||
      elements > kMaxInputElements) {
    return;
  }

  std::vector<uint8_t> input_bytes = fuzzing::MakeValues(input_type, elements, 37);
  fuzzing::OverlayBytes(input_data, &input_bytes);
  fuzzing::ApplyCentralTensorInputInvariants(input_type, &input_bytes);

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};
  const auto input_shape = builder.CreateVector(shape);
  const auto output_shape = builder.CreateVector(shape);
  const auto input_tensor = CreateTensor(builder, input_shape, input_type, 0);
  const auto output_tensor = CreateTensor(builder, output_shape, output_type);
  const auto cast_options =
      CreateCastOptions(builder, input_type, output_type).Union();

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "cast_fuzz";
  model_spec.builtin_operator = BuiltinOperator_CAST;
  model_spec.version = 9;
  model_spec.builtin_options_type = BuiltinOptions_CastOptions;
  model_spec.builtin_options = cast_options;
  model_spec.tensors = {input_tensor, output_tensor};
  model_spec.buffers = std::move(buffers);
  model_spec.model_inputs = {0};
  model_spec.model_outputs = {1};
  model_spec.op_inputs = {0};
  model_spec.op_outputs = {1};

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = ops::builtin::Register_CAST();
  run_spec.min_version = 1;
  run_spec.max_version = 9;
  run_spec.max_live_allocation_bytes = kMaxLiveAllocationBytes;
  run_spec.runtime_tensors.push_back({0, shape, std::move(input_bytes)});
  run_spec.invoke = true;

  (void)fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

}  // namespace
}  // namespace tflite

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  FuzzedDataProvider fdp(data, size);
  tflite::RunCast(fdp);
  return 0;
}
