#include <cstddef>
#include <cstdint>

#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/verifier.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  (void)litert::VerifyModel(data, size);
  litert::BufferRef<uint8_t> buffer(data, size);
  (void)litert::internal::LoadModelFromBuffer(buffer);
  return 0;
}
