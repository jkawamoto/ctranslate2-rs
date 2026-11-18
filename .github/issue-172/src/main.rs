fn main() {
    let mut config = cmake::Config::new(".github/issue-172");
    config
        .profile("Release")
        .cxxflag("/EHsc")
        .static_crt(true)
        .define("CT2_SOURCE", std::env::var("CT2_SOURCE").unwrap())
        .define("CMAKE_POLICY_VERSION_MINIMUM", "3.5")
        .build_target("probe");
    match std::env::var("MODE").unwrap().as_str() {
        "optimized" => {
            config.define("CMAKE_CXX_FLAGS_RELEASE", "/O2 /Ob2 /DNDEBUG");
        }
        "explicit-generator" => {
            config.generator("Visual Studio 17 2022");
        }
        "no-dispatch" => {
            config.define("ENABLE_CPU_DISPATCH", "OFF");
        }
        "default" => {}
        mode => panic!("Unexpected mode: {mode}"),
    }
    config.build();
}
