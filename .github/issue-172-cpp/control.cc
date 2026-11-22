#include <iostream>
#include <string>
#include <intrin.h>
#include <immintrin.h>

int main(int argc, char* argv[]) {
  std::cout << std::unitbuf;
  int cpuinfo[4] = {0};
  __cpuid(cpuinfo, 1);
  bool sse4_1 = (cpuinfo[2] & (1 << 19)) != 0;
  bool avx = (cpuinfo[2] & (1 << 28)) != 0;
  bool osxsave = (cpuinfo[2] & (1 << 27)) != 0;

  std::cout << "Control CPUID: SSE4.1=" << sse4_1
            << " AVX=" << avx
            << " OSXSAVE=" << osxsave << std::endl;

  if (argc > 1 && std::string(argv[1]) == "--trigger-avx") {
    std::cout << "Executing explicit AVX instruction (_mm256_zeroall)..." << std::endl;
    _mm256_zeroall();
    std::cout << "AVX instruction executed successfully." << std::endl;
    return 0;
  }

  std::cout << "Control check passed." << std::endl;
  return 0;
}
