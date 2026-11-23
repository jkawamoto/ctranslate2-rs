#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__cpuid, _mm256_zeroall};

fn main() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let cpuid1 = __cpuid(1);
        let ecx = cpuid1.ecx;
        let sse4_1 = (ecx & (1 << 19)) != 0;
        let osxsave = (ecx & (1 << 27)) != 0;
        let avx = (ecx & (1 << 28)) != 0;

        println!("Control CPUID: SSE4.1={} AVX={} OSXSAVE={}", sse4_1, avx, osxsave);

        let args: Vec<String> = std::env::args().collect();
        if args.len() > 1 && args[1] == "--trigger-avx" {
            println!("Executing explicit AVX instruction (_mm256_zeroall)...");
            _mm256_zeroall();
            println!("AVX instruction executed successfully.");
            return;
        }

        println!("Control check passed.");
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("Not an x86_64 architecture.");
    }
}
