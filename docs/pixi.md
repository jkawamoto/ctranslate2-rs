# Using Pixi & direnv

> You must have an NVIDIA GPU on your machine and you must install the CUDA drivers. The CUDA driver cannot be installed with conda and must be installed on your system using an appropriate installation method. [Reference](https://conda-forge.org/docs/maintainer/knowledge_base/#prerequisites).

- [Pixi Homepage](https://pixi.prefix.dev/latest/)
- [direnv](https://direnv.net/)

A development environment with `pixi` and `direnv` is very useful because it provides a self-contained, reproducible native build toolchain without requiring root privileges or polluting host system directories:

- **Isolated native & CUDA toolchain:** It pins and installs native dependencies (CMake, Ninja, compilers, CUDA nvcc, and cuDNN) alongside Rust in a local `.pixi/` directory; the optional `platform` environment adds NCCL and OpenMPI.
- **Seamless shell & editor integration:** With `direnv`, all environment variables (`$PATH`, `$CONDA_PREFIX`, `$CUDA_TOOLKIT_ROOT_DIR`, and compiler flags) are automatically activated whenever you enter the project directory. This ensures IDEs and language servers (like `rust-analyzer` in Zed or VS Code) immediately find the correct compilers and headers without extra wrapper scripts.
- **Deterministic builds:** The `pixi.lock` file guarantees that every contributor and CI pipeline builds against identical versions of native C++ and CUDA libraries, eliminating "works on my machine" inconsistencies.

> [!tip]
> [direnv](https://direnv.net/)
>
> ```bash
> pixi global install direnv
> ```
>
> ```bash
> if [ -d "$HOME/.pixi/bin" ] ; then
>     export PATH="$HOME/.pixi/bin:$PATH"
> fi
> ```

# Setting up the environment

After cloning the repository:

```bash
cd ctranslate2-rs

pixi install

direnv allow
```

> [!tip]
> You can check with:
>
> ```bash
> # must be the same as `pixi run shell-hook`
> env
> ```

# Pixi Issues

Move from full `cuda` (`conda-forge`) to minimal dependencies:

```toml
[dependencies]
cuda = ">=12.9,<13.0"
```

```toml
[dependencies]
cuda-version = "12.9.*"
cudnn = "8.*"
cuda-nvcc = "*"
libcublas-dev = "*"
```

## NCCL and MPI > tensor-parallel

[NCCL](https://developer.nvidia.com/nccl)

CTranslate2 requires `NCCL` and `MPI` when building with tensor parallelism (see lines 500–502).

```CmakeLists.txt
if (WITH_TENSOR_PARALLEL)
  find_package(MPI REQUIRED)
  find_package(NCCL REQUIRED)
```

```toml
[environments]
default = []
platform = ["tensor-parallel", "hub"]

[feature.tensor-parallel.dependencies]
openmpi = "*"
nccl = "*"
```

## NVIDIA Headers

NVIDIA distributes the CUDA Toolkit as modular redistributable packages. NVIDIA uses a target-architecture layout to support cross-compilation:

```text
$CONDA_PREFIX/targets/
  └── x86_64-linux/
      ├── include/     <-- cuda.h, cuda_runtime.h, etc.
      └── lib/         <-- libcudart.so, etc.
```

Conda-forge packages NVIDIA's official redistributables directly using NVIDIA's layout. To make compilers find it, conda-forge's `cuda-nvcc` provides an activation script (`~cuda-nvcc_activate.sh`) that automatically adds:

```bash
-I$CONDA_PREFIX/targets/x86_64-linux/include
-L$CONDA_PREFIX/targets/x86_64-linux/lib
```

```toml
[activation.env]
CFLAGS = "$CFLAGS -pthread"
CXXFLAGS = "$CXXFLAGS -pthread"
CUDA_ARCH_LIST = "Auto"
CUDA_TOOLKIT_ROOT_DIR = "$CONDA_PREFIX/targets/x86_64-linux"
CUDA_PATH = "$CONDA_PREFIX/targets/x86_64-linux"
```
