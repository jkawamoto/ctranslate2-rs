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
  if (!condition)
    throw std::runtime_error(message);
}

int main() {
  using namespace ctranslate2;
  using cpu::CpuIsa;
  std::cout << std::unitbuf;
  cpu::set_num_threads(1);
  int registers[4];
  __cpuid(registers, 1);
  std::cout << "avx_cpuid=" << bool(registers[2] & (1 << 28))
            << " isa=" << cpu::isa_to_str(cpu::get_cpu_isa()) << '\n';
  const float x[] = {-2, -1, 0, 1, 2};
  float y[5];
  std::cout << "generic reduce_sum\n";
  check(cpu::reduce_sum<CpuIsa::GENERIC>(x, 5) == 0, "reduce_sum");
  std::cout << "dispatched sum\n";
  check(primitives<>::sum(x, 5) == 0, "sum");
  std::cout << "generic gelu\n";
  cpu::gelu<CpuIsa::GENERIC>(x, y, 5);
  check(std::abs(y[3] - .8413447f) < .0001f, "gelu");
  std::cout << "generic layer_norm\n";
  const float gamma[] = {1, 1, 1, 1, 1}, beta[] = {0, 0, 0, 0, 0};
  cpu::layer_norm<CpuIsa::GENERIC>(x, gamma, beta, y, 1, 5, 1e-5f);
  check(std::abs(y[3] - 1 / std::sqrt(2.f + 1e-5f)) < .0001f, "layer_norm");
  std::cout << "generic softmax\n";
  cpu::softmax<CpuIsa::GENERIC>(x, nullptr, y, 1, 5, false);
  float sum = 0;
  for (float value : y)
    sum += value;
  check(std::abs(sum - 1) < .0001f, "softmax");
  std::cout << "gemm backend="
            << cpu::gemm_backend_to_str(cpu::get_gemm_backend(ComputeType::FLOAT32)) << '\n';
  for (int n : {2, 65}) {
    std::vector<float> a(n * n, 1), b(n * n, 2), c(n * n, 0);
    primitives<>::gemm(false, false, false, false, n, n, n,
                       1.f, a.data(), n, b.data(), n, 0.f, c.data(), n);
    for (float value : c)
      check(std::abs(value - 2 * n) < .001f, "gemm");
  }
  std::cout << "PASS\n";
}
