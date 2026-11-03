// Known-answer checks over the TFLite runtime and the LiteRT model loader/verifier.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/verifier.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/version.h"

namespace {

int g_passed = 0;
int g_failed = 0;

void Check(bool ok, const char* name) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
  (ok ? g_passed : g_failed)++;
}

std::string ReadFile(const char* path) {
  std::ifstream f(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

std::unique_ptr<tflite::Interpreter> Build(const void* buf, size_t len) {
  auto model = tflite::FlatBufferModel::BuildFromBuffer(
      static_cast<const char*>(buf), len);
  if (!model) return nullptr;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interp;
  if (tflite::InterpreterBuilder(*model, resolver)(&interp) != kTfLiteOk) {
    return nullptr;
  }
  if (interp->AllocateTensors() != kTfLiteOk) return nullptr;
  // The interpreter does not own the model buffer; keep a copy alive.
  static std::vector<std::unique_ptr<tflite::FlatBufferModel>> keep;
  keep.push_back(std::move(model));
  return interp;
}

struct OneOp {
  tflite::BuiltinOperator op;
  tflite::TensorType in_type, out_type;
  int num_inputs;
  std::vector<int> shape;
  tflite::BuiltinOptions opt_type = tflite::BuiltinOptions_NONE;
  flatbuffers::Offset<void> options = 0;
};

std::string BuildOneOpModel(flatbuffers::FlatBufferBuilder& b, OneOp o) {
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  std::vector<int32_t> inputs, op_inputs;
  for (int i = 0; i < o.num_inputs; ++i) {
    tensors.push_back(tflite::CreateTensor(b, b.CreateVector(o.shape), o.in_type, 0));
    inputs.push_back(i);
  }
  tensors.push_back(tflite::CreateTensor(b, b.CreateVector(o.shape), o.out_type, 0));
  std::vector<int32_t> outputs = {o.num_inputs};
  auto code = tflite::CreateOperatorCode(b, 0, 0, 1, o.op);
  auto op = tflite::CreateOperator(b, 0, b.CreateVector(inputs),
                                   b.CreateVector(outputs), o.opt_type, o.options);
  auto sg = tflite::CreateSubGraph(b, b.CreateVector(tensors),
                                   b.CreateVector(inputs), b.CreateVector(outputs),
                                   b.CreateVector(&op, 1));
  auto buf = tflite::CreateBuffer(b);
  auto m = tflite::CreateModel(b, TFLITE_SCHEMA_VERSION, b.CreateVector(&code, 1),
                               b.CreateVector(&sg, 1), b.CreateString("oracle"),
                               b.CreateVector(&buf, 1));
  tflite::FinishModelBuffer(b, m);
  return std::string(reinterpret_cast<const char*>(b.GetBufferPointer()), b.GetSize());
}

void TestAddModel(const std::string& add_bin) {
  auto interp = Build(add_bin.data(), add_bin.size());
  Check(interp != nullptr, "add.bin builds");
  if (!interp) return;
  float* in = interp->typed_input_tensor<float>(0);
  const int n = 8 * 8 * 3;
  for (int i = 0; i < n; ++i) in[i] = 0.5f * i - 3.0f;
  Check(interp->Invoke() == kTfLiteOk, "add.bin invokes");
  const float* out = interp->typed_output_tensor<float>(0);
  bool ok = true;
  for (int i = 0; i < n; ++i) ok = ok && out[i] == 3.0f * (0.5f * i - 3.0f);
  Check(ok, "add.bin computes 3*x (x+x+x) for a known input");
}

void TestBinaryFloat(const char* name, tflite::BuiltinOperator op,
                     tflite::BuiltinOptions ot,
                     flatbuffers::Offset<void> (*mk)(flatbuffers::FlatBufferBuilder&),
                     const float* expected) {
  flatbuffers::FlatBufferBuilder b;
  OneOp o{op, tflite::TensorType_FLOAT32, tflite::TensorType_FLOAT32, 2, {4}, ot, mk(b)};
  std::string m = BuildOneOpModel(b, o);
  auto interp = Build(m.data(), m.size());
  Check(interp != nullptr, (std::string(name) + " builds").c_str());
  if (!interp) return;
  const float a[4] = {1.5f, -2.0f, 8.0f, 0.25f};
  const float c[4] = {4.0f, 0.5f, -0.5f, 16.0f};
  std::memcpy(interp->typed_input_tensor<float>(0), a, sizeof(a));
  std::memcpy(interp->typed_input_tensor<float>(1), c, sizeof(c));
  bool ok = interp->Invoke() == kTfLiteOk;
  const float* out = interp->typed_output_tensor<float>(0);
  for (int i = 0; ok && i < 4; ++i) ok = out[i] == expected[i];
  Check(ok, name);
}

void TestCast() {
  flatbuffers::FlatBufferBuilder b;
  OneOp o{tflite::BuiltinOperator_CAST, tflite::TensorType_FLOAT32,
          tflite::TensorType_INT32, 1, {6}, tflite::BuiltinOptions_CastOptions,
          tflite::CreateCastOptions(b, tflite::TensorType_FLOAT32,
                                    tflite::TensorType_INT32).Union()};
  std::string m = BuildOneOpModel(b, o);
  auto interp = Build(m.data(), m.size());
  Check(interp != nullptr, "cast model builds");
  if (!interp) return;
  const float in[6] = {-2.7f, -0.5f, 0.5f, 2.9f, 100.0f, 7.0f};
  std::memcpy(interp->typed_input_tensor<float>(0), in, sizeof(in));
  bool ok = interp->Invoke() == kTfLiteOk;
  const int32_t expected[6] = {-2, 0, 0, 2, 100, 7};
  const int32_t* out = interp->typed_output_tensor<int32_t>(0);
  for (int i = 0; ok && i < 6; ++i) ok = out[i] == expected[i];
  Check(ok, "cast float32->int32 truncates toward zero");
}

void TestSoftmax() {
  flatbuffers::FlatBufferBuilder b;
  OneOp o{tflite::BuiltinOperator_SOFTMAX, tflite::TensorType_FLOAT32,
          tflite::TensorType_FLOAT32, 1, {1, 4}, tflite::BuiltinOptions_SoftmaxOptions,
          tflite::CreateSoftmaxOptions(b, 1.0f).Union()};
  std::string m = BuildOneOpModel(b, o);
  auto interp = Build(m.data(), m.size());
  Check(interp != nullptr, "softmax model builds");
  if (!interp) return;
  const float in[4] = {0.0f, 0.0f, std::log(2.0f), std::log(4.0f)};
  std::memcpy(interp->typed_input_tensor<float>(0), in, sizeof(in));
  bool ok = interp->Invoke() == kTfLiteOk;
  const float expected[4] = {1.0f / 8, 1.0f / 8, 2.0f / 8, 4.0f / 8};
  const float* out = interp->typed_output_tensor<float>(0);
  for (int i = 0; ok && i < 4; ++i) ok = std::fabs(out[i] - expected[i]) < 1e-5f;
  Check(ok, "softmax([0,0,ln2,ln4]) == [1/8,1/8,2/8,4/8]");
}

void TestLiteRt(const std::string& add_bin) {
  Check(litert::VerifyModel(add_bin.data(), add_bin.size()).ok(),
        "litert verifier accepts add.bin");
  Check(!litert::VerifyModel(add_bin.data(), add_bin.size() / 2).ok(),
        "litert verifier rejects a truncated add.bin");
  const std::string junk(256, '\xAB');
  Check(!litert::VerifyModel(junk.data(), junk.size()).ok(),
        "litert verifier rejects junk bytes");

  litert::BufferRef<uint8_t> ref(add_bin.data(), add_bin.size());
  auto model = litert::internal::LoadModelFromBuffer(ref);
  Check(model.HasValue(), "litert loads add.bin");
  if (!model.HasValue()) return;
  const auto& sg = (*model)->Subgraph(0);
  bool ok = (*model)->NumSubgraphs() == 1 && sg.Ops().size() == 2 &&
            sg.Op(0).OpCode() == kLiteRtOpCodeTflAdd &&
            sg.Op(1).OpCode() == kLiteRtOpCodeTflAdd &&
            sg.Inputs().size() == 1 && sg.Outputs().size() == 1;
  Check(ok, "litert add.bin graph: 1 input, 1 output, two TFL_ADD ops");
}

}  // namespace

int main() {
  const std::string add_bin = ReadFile("tflite/testdata/add.bin");
  Check(!add_bin.empty(), "read tflite/testdata/add.bin");
  if (add_bin.empty()) return 1;

  TestAddModel(add_bin);
  static const float kMul[4] = {6.0f, -1.0f, -4.0f, 4.0f};
  static const float kSub[4] = {-2.5f, -2.5f, 8.5f, -15.75f};
  TestBinaryFloat("mul float32 known answers", tflite::BuiltinOperator_MUL,
                  tflite::BuiltinOptions_MulOptions,
                  [](flatbuffers::FlatBufferBuilder& b) {
                    return tflite::CreateMulOptions(b).Union();
                  },
                  kMul);
  TestBinaryFloat("sub float32 known answers", tflite::BuiltinOperator_SUB,
                  tflite::BuiltinOptions_SubOptions,
                  [](flatbuffers::FlatBufferBuilder& b) {
                    return tflite::CreateSubOptions(b).Union();
                  },
                  kSub);
  TestCast();
  TestSoftmax();
  TestLiteRt(add_bin);

  std::printf("ORACLE passed=%d failed=%d\n", g_passed, g_failed);
  return g_failed == 0 ? 0 : 1;
}
