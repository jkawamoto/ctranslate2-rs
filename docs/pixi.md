# Using Pixi & direnv

> You must have an NVIDIA GPU on your machine and you must install the CUDA drivers. The CUDA driver cannot be installed with conda and must be installed on your system using an appropriate installation method. [Reference](https://conda-forge.org/docs/maintainer/knowledge_base/#prerequisites).

- [Pixi Homepage](https://pixi.prefix.dev/latest/)
- [direnv](https://direnv.net/)

A development environment with `pixi` and `direnv` is very useful because it provides a self-contained, reproducible native build toolchain without requiring root privileges or polluting host system directories:

- **Isolated native & CUDA toolchain:** Pixi installs Rust, CMake, Ninja, the C/C++ compilers, CUDA (`nvcc`, cuBLAS) and cuDNN into the project's `.pixi/` directory, without root privileges or changes to system directories.
- **Environments per feature set:** `default` covers building `ct2rs`; `hub` adds OpenSSL for the `hub` feature; `platform` adds NCCL and OpenMPI for `tensor-parallel` (still incomplete).
- **Shell & editor integration:** `direnv` activates the environment (`$PATH`, `$CONDA_PREFIX`, `$CUDA_TOOLKIT_ROOT_DIR`, compiler flags) whenever you enter the project directory. Zed picks it up natively; VS Code needs the direnv extension (`mkhl.direnv`). For rust-analyzer, see [Rust-analyzer LSP](#rust-analyzer-lsp).

> [!tip]
> [direnv](https://direnv.net/)
>
> ```bash
> pixi global install direnv
> ```
>
> Put this in `~/.profile`, not `~/.bashrc`: `.profile` is read at login, so desktop-launched apps like Zed or VS Code also get `~/.pixi/bin` on their `PATH` and can find `direnv` and the other tools installed with `pixi global`.
>
> ```bash
> if [ -d "$HOME/.pixi/bin" ] ; then
>     export PATH="$HOME/.pixi/bin:$PATH"
> fi
> ```

## Setting up the environment

After cloning the repository:

```bash
cd ctranslate2-rs

pixi install -a

direnv allow
```

> [!tip]
> You can check with:
>
> ```bash
> # must be the same as `pixi run shell-hook`
> env
> ```

## Rust-analyzer LSP

By default, rust-analyzer runs `cargo check --workspace`, which ignores `default-members` and builds `ct2rs-platform`, so CMake fails on the missing MPI/NCCL dependencies of `tensor-parallel`.

To keep the editor working in the `default` Pixi environment, limit rust-analyzer's build scripts and checks to `ct2rs`:

- `.zed/settings.json`:
  ```json
  {
      "lsp": {
          "rust-analyzer": {
              "initialization_options": {
                  "cargo": {
                      "buildScripts": {
                          "overrideCommand": [
                              "cargo", "check", "--quiet",
                              "--package", "ct2rs",
                              "--message-format=json",
                              "--all-targets", "--keep-going"
                          ]
                      }
                  },
                  "check": {
                      "workspace": false,
                      "extraArgs": ["--package", "ct2rs"]
                  }
              }
          }
      }
  }
  ```

- `.vscode/settings.json`:
  ```json
  {
      "rust-analyzer.cargo.buildScripts.overrideCommand": [
          "cargo",
          "check",
          "--quiet",
          "--package",
          "ct2rs",
          "--message-format=json",
          "--all-targets",
          "--keep-going"
      ],
      "rust-analyzer.check.workspace": false,
      "rust-analyzer.check.extraArgs": ["--package", "ct2rs"]
  }
  ```

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
