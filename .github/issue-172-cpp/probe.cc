#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <intrin.h>

#include "cpu/kernels.h"
#include "cpu/backend.h"
#include "cpu/parallel.h"
#include "ctranslate2/primitives.h"

void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED check: " << message << std::endl;
    throw std::runtime_error(message);
  }
}

int main() {
  using namespace ctranslate2;
  using cpu::CpuIsa;

  std::cout << std::unitbuf;
  std::cerr << std::unitbuf;

  std::cout << "=== PROBE START ===" << std::endl;

  cpu::set_num_threads(1);

  // 1. CPUID
  int registers[4] = {0};
  __cpuid(registers, 1);
  bool sse4_1_cpuid = (registers[2] & (1 << 19)) != 0;
  bool avx_cpuid = (registers[2] & (1 << 28)) != 0;
  std::cout << "probe cpuid: sse4_1=" << sse4_1_cpuid
            << " avx=" << avx_cpuid << std::endl;

  // 2. CTranslate2 ISA detection
  std::cout << "probe ctranslate2 isa=" << cpu::isa_to_str(cpu::get_cpu_isa()) << std::endl;
  std::cout << "ct2 cpu_supports: GENERIC=" << cpu::cpu_supports(CpuIsa::GENERIC)
            << " SSE41=" << cpu::cpu_supports(CpuIsa::SSE41)
            << " AVX=" << cpu::cpu_supports(CpuIsa::AVX)
            << " AVX2=" << cpu::cpu_supports(CpuIsa::AVX2)
            << " AVX512=" << cpu::cpu_supports(CpuIsa::AVX512)
            << std::endl;

  const float x[] = {-2.f, -1.f, 0.f, 1.f, 2.f};
  float y[5] = {0.f};

  // 3. Direct GENERIC kernel calls
  std::cout << "[Step 1] calling cpu::reduce_sum<CpuIsa::GENERIC>..." << std::endl;
  float sum_res = cpu::reduce_sum<CpuIsa::GENERIC>(x, 5);
  std::cout << "  reduce_sum result: " << sum_res << std::endl;
  check(sum_res == 0.f, "reduce_sum");

  std::cout << "[Step 2] calling primitives<>::sum (runtime dispatch)..." << std::endl;
  float sum_prim = primitives<>::sum(x, 5);
  std::cout << "  primitives::sum result: " << sum_prim << std::endl;
  check(sum_prim == 0.f, "primitives sum");

  std::cout << "[Step 3] calling cpu::gelu<CpuIsa::GENERIC>..." << std::endl;
  cpu::gelu<CpuIsa::GENERIC>(x, y, 5);
  check(std::abs(y[3] - 0.8413447f) < 0.0001f, "gelu");

  std::cout << "[Step 4] calling cpu::layer_norm<CpuIsa::GENERIC> (calls sqrt and max)..." << std::endl;
  const float gamma[] = {1.f, 1.f, 1.f, 1.f, 1.f};
  const float beta[] = {0.f, 0.f, 0.f, 0.f, 0.f};
  cpu::layer_norm<CpuIsa::GENERIC>(x, gamma, beta, y, 1, 5, 1e-5f);
  check(std::abs(y[3] - 1.f / std::sqrt(2.f + 1e-5f)) < 0.0001f, "layer_norm");

  std::cout << "[Step 5] calling cpu::softmax<CpuIsa::GENERIC>..." << std::endl;
  cpu::softmax<CpuIsa::GENERIC>(x, nullptr, y, 1, 5, false);
  float sum_val = 0.f;
  for (float v : y) sum_val += v;
  check(std::abs(sum_val - 1.f) < 0.0001f, "softmax");

  // 4. Direct call to std::max and std::sqrt
  std::cout << "[Step 6] calling std::max<float> and std::sqrt(float) directly..." << std::endl;
  float m1 = 1.5f, m2 = 2.5f;
  const float& mres = std::max(m1, m2);
  check(mres == 2.5f, "std::max");
  float sqres = std::sqrt(4.f);
  check(sqres == 2.f, "std::sqrt");

  // 5. GEMM (Ruy)
  std::cout << "[Step 7] calling primitives<>::gemm (Ruy backend)..." << std::endl;
  std::cout << "  gemm backend="
            << cpu::gemm_backend_to_str(cpu::get_gemm_backend(ComputeType::FLOAT32)) << std::endl;
  for (int n : {2, 65}) {
    std::vector<float> a(n * n, 1.f), b(n * n, 2.f), c(n * n, 0.f);
    primitives<>::gemm(false, false, false, false, n, n, n,
                       1.f, a.data(), n, b.data(), n, 0.f, c.data(), n);
    for (float val : c)
      check(std::abs(val - 2.f * n) < 0.001f, "gemm");
  }

  std::cout << "=== PROBE ALL TESTS PASSED ===" << std::endl;
  return 0;
}
