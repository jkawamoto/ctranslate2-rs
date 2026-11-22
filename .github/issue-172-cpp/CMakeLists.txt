cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0091 NEW)

# MSVC 静的 CRT (/MT)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

project(issue_172_cpp_repro CXX)
set(CMAKE_CXX_STANDARD 17)

# CTranslate2 の設定
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(BUILD_CLI OFF CACHE BOOL "")
set(WITH_MKL OFF CACHE BOOL "")
set(WITH_RUY ON CACHE BOOL "")
set(OPENMP_RUNTIME NONE CACHE STRING "")

# CTranslate2 サブディレクトリを追加
add_subdirectory("${CT2_SOURCE}" ct2)

# 1. 対照プログラム control
add_executable(control control.cc)
target_compile_options(control PRIVATE /arch:AVX /O2 /DNDEBUG)
target_link_options(control PRIVATE /DEBUG "/MAP:${CMAKE_CURRENT_BINARY_DIR}/control.map")

# 共通設定マクロ
macro(configure_probe target_name)
  add_executable(${target_name} probe.cc)
  target_include_directories(${target_name} PRIVATE "${CT2_SOURCE}/src" "${CT2_SOURCE}/third_party")
  target_compile_definitions(${target_name} PRIVATE CT2_X86_BUILD)
  target_compile_options(${target_name} PRIVATE /O2 /Ob2 /DNDEBUG)
  target_link_options(${target_name} PRIVATE /DEBUG "/MAP:${CMAKE_CURRENT_BINARY_DIR}/${target_name}.map" /VERBOSE:LIB)
endmacro()

# 2. 通常リンク (Normal link: target_link_libraries で ctranslate2 静的ライブラリのみリンク)
configure_probe(probe_normal)
target_link_libraries(probe_normal PRIVATE ctranslate2)

# 3. CPU Dispatch が有効な場合のみの対照ターゲット
if(ENABLE_CPU_DISPATCH)
  set(KERNEL_AVX512_OBJ "${CMAKE_CURRENT_BINARY_DIR}/ct2/ctranslate2.dir/$<CONFIG>/kernels_avx512.obj")
  set(KERNEL_AVX_OBJ    "${CMAKE_CURRENT_BINARY_DIR}/ct2/ctranslate2.dir/$<CONFIG>/kernels_avx.obj")
  set(KERNEL_GENERIC_OBJ "${CMAKE_CURRENT_BINARY_DIR}/ct2/ctranslate2.dir/$<CONFIG>/kernels.obj")

  # 3a. ISA 別オブジェクト先行リンク (kernels_avx512.obj を ctranslate2 より前)
  configure_probe(probe_avx512_first)
  add_dependencies(probe_avx512_first ctranslate2)
  target_link_libraries(probe_avx512_first PRIVATE "${KERNEL_AVX512_OBJ}" ctranslate2)

  # 3b. ISA 別オブジェクト先行リンク (kernels_avx.obj を ctranslate2 より前)
  configure_probe(probe_avx_first)
  add_dependencies(probe_avx_first ctranslate2)
  target_link_libraries(probe_avx_first PRIVATE "${KERNEL_AVX_OBJ}" ctranslate2)

  # 3c. GENERIC オブジェクト先行リンク (kernels.obj を ctranslate2 より前)
  configure_probe(probe_generic_first)
  add_dependencies(probe_generic_first ctranslate2)
  target_link_libraries(probe_generic_first PRIVATE "${KERNEL_GENERIC_OBJ}" ctranslate2)

  # 3d. /WHOLEARCHIVE:ctranslate2
  configure_probe(probe_wholearchive)
  target_link_libraries(probe_wholearchive PRIVATE ctranslate2)
  target_link_options(probe_wholearchive PRIVATE "/WHOLEARCHIVE:ctranslate2")

  # 3e. /FORCE:MULTIPLE (AVX512 オブジェクト先行 + /FORCE:MULTIPLE)
  configure_probe(probe_force_multiple)
  add_dependencies(probe_force_multiple ctranslate2)
  target_link_libraries(probe_force_multiple PRIVATE "${KERNEL_AVX512_OBJ}" ctranslate2)
  target_link_options(probe_force_multiple PRIVATE "/FORCE:MULTIPLE")
endif()
