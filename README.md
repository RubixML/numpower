<p align="center">
  <img src="https://github.com/user-attachments/assets/11bc8f07-60cb-4628-bc88-5dee4b22bdc6" width="200" height="200">
</p>

<h1 align="center">Rubix NumPower</h1>

**Rubix NumPower** is a PHP extension for fast math on large arrays — think NumPy, but for PHP. It gives you N-dimensional arrays (`NDArray`) and operations on them that run **tens to thousands of times faster** than the same code written with plain PHP arrays.

Under the hood NumPower uses the same battle-tested numerical libraries (OpenBLAS, LAPACK) that NumPy, SciPy, and most machine-learning frameworks rely on, plus AVX2 instructions on your CPU and — if you have an NVIDIA GPU — CUDA kernels for matrix operations that don't fit on the CPU comfortably.

**Typical use cases:** machine learning, neural networks, signal processing, image processing, statistics, scientific computing — anything where you need to multiply, add, slice, or transform big matrices and vectors.

---

## Table of contents

- [What runs where](#what-runs-where)
- [Dependencies](#dependencies)
  - [Required](#required)
  - [Optional](#optional)
- [Installation](#installation)
  - [Ubuntu / Debian](#ubuntu--debian)
  - [Fedora / RHEL / Rocky / Alma](#fedora--rhel--rocky--alma)
  - [Arch Linux / Manjaro](#arch-linux--manjaro)
  - [macOS](#macos)
  - [Windows](#windows)
- [GPU support](#gpu-support)
- [Verifying the installation](#verifying-the-installation)
- [Using the GPU from your code](#using-the-gpu-from-your-code)

---

## What runs where

| Platform | CPU build | GPU (CUDA) build |
|---|---|---|
| **Linux** (Ubuntu, Debian, Fedora, Arch, …) | Yes | Yes — needs an NVIDIA card and the CUDA Toolkit |
| **macOS** (Apple Silicon and Intel) | Yes | No — Apple dropped CUDA support years ago |
| **Windows** (10/11, Server 2022/2025) | Yes | Yes — needs an NVIDIA card and the CUDA Toolkit |

You can build NumPower for CPU-only and add GPU support later by re-running the build with the right flags.

---

## Dependencies

### Required

You need these to build NumPower on any platform:

- **PHP 8.1 or newer** with its development headers (so `phpize` and `php-config` are available).
- A **C / C++ compiler**: GCC or Clang on Linux/macOS, Visual Studio (or Build Tools) on Windows.
- **autoconf** — only on Linux and macOS; Windows uses its own build flow.
- **OpenBLAS** and **LAPACK/LAPACKE** — the numerical libraries NumPower delegates matrix math to. Installed via your system package manager (`apt`, `dnf`, `pacman`, Homebrew) or, on Windows, downloaded as a prebuilt ZIP.

That's it. With those installed, `phpize → ./configure → make → make install` will give you a working CPU build.

### Optional

Install these only if you need the feature they enable. The build auto-detects whatever you have installed; the rest is silently skipped.

| Feature you want | What to install | Notes |
|---|---|---|
| **Run code on your NVIDIA GPU** | NVIDIA **CUDA Toolkit** (gives you `nvcc`, `cublas`, `cudart`) | Available for Linux and Windows. After installing, rebuild NumPower with `--with-cuda`. |
| **Train / run CNN models on the GPU** | **cuDNN** (download from NVIDIA's site) | On top of the CUDA Toolkit. Enables `Conv1D` / `Conv2D` on the GPU. Skip if you don't use convolutional networks. |
| **Faster CPU math on Intel chips** | **Intel MKL** | A drop-in replacement for OpenBLAS, typically 2–3× faster on modern Intel CPUs. The build picks it up automatically when it's on the linker path. |
| **Maximum precision for the `float128` data type** | **libquadmath** (Linux + GCC only) | Without it `float128` works but with slightly less precision (~32 decimal digits instead of ~34). Most users never notice. |
| **Load and save images directly as NDArrays** | **PHP-GD extension** (plus `libjpeg`, `libpng`, `libwebp` if missing) | Useful for computer-vision workflows. Without it you can still pass PHP arrays around the GD calls. |
| **AVX2 / SSE CPU vector instructions** | Nothing — auto-detected | Already there on any x86-64 CPU made since roughly 2013. On Apple Silicon and other ARM64 machines NumPower falls back to scalar code automatically. |

---

## Installation

The steps below build NumPower from source against the PHP that's already on your `PATH`. If you juggle several PHP versions, use the absolute path to the one you want (e.g. `/usr/bin/phpize8.3`) when running `phpize`.

### Ubuntu / Debian

**1. Install the required packages**

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    autoconf \
    php-dev \
    libopenblas-dev \
    liblapacke-dev
```

If you target a specific PHP version, replace `php-dev` with the matching package — for example `php8.3-dev`.

**2. (Optional) Install extras**

```bash
# GPU support (NVIDIA card required):
sudo apt-get install -y nvidia-cuda-toolkit
# cuDNN is a separate download from https://developer.nvidia.com/cudnn

# Image bridge with GD:
sudo apt-get install -y php-gd libjpeg-dev libpng-dev libwebp-dev
```

**3. Build and install**

```bash
git clone https://github.com/RubixML/numpower.git
cd numpower
phpize
./configure                # add --with-cuda if you installed the CUDA Toolkit
make -j$(nproc)
sudo make install          # use `sudo make install-cuda` for GPU builds
```

**4. Turn the extension on**

```bash
echo "extension=ndarray.so" | sudo tee /etc/php/$(php -r 'echo PHP_MAJOR_VERSION.".".PHP_MINOR_VERSION;')/mods-available/ndarray.ini
sudo phpenmod ndarray
```

Done. Test it with `php -m | grep ndarray`.

### Fedora / RHEL / Rocky / Alma

```bash
sudo dnf install -y gcc gcc-c++ make autoconf php-devel openblas-devel lapack-devel
```

Optional:

```bash
sudo dnf install -y cuda-toolkit                       # GPU (from NVIDIA's RPM repo)
sudo dnf install -y php-gd libjpeg-turbo-devel libpng-devel libwebp-devel   # image bridge
```

Then run the same `phpize → ./configure → make -j → sudo make install` as above, and add `extension=ndarray.so` to `/etc/php.d/40-ndarray.ini`.

### Arch Linux / Manjaro

```bash
sudo pacman -S --needed base-devel autoconf php openblas lapacke
```

Optional:

```bash
sudo pacman -S cuda cudnn      # GPU
sudo pacman -S php-gd          # image bridge
```

Build with the standard PHP extension flow, then add `extension=ndarray.so` to `/etc/php/conf.d/ndarray.ini`.

### macOS

**CUDA is not available on macOS** — Apple dropped CUDA support a few years ago — so the build is CPU-only. The `float128` data type still works, just with slightly reduced precision (no `libquadmath` on Apple Clang).

**1. Install dependencies via Homebrew**

```bash
brew update
brew install php openblas lapack autoconf
```

If you want a specific PHP version, use `brew install php@8.3` and put it ahead of the default `php` on your `PATH`.

**2. Build NumPower**

Homebrew installs OpenBLAS and LAPACK in its own folders that the compiler doesn't search by default, so we point the build at them with a few environment variables:

```bash
git clone https://github.com/RubixML/numpower.git
cd numpower

OPENBLAS_PREFIX=$(brew --prefix openblas)
LAPACK_PREFIX=$(brew --prefix lapack)
export CPPFLAGS="-I${OPENBLAS_PREFIX}/include -I${LAPACK_PREFIX}/include ${CPPFLAGS}"
export LDFLAGS="-L${OPENBLAS_PREFIX}/lib -L${LAPACK_PREFIX}/lib ${LDFLAGS}"
export PKG_CONFIG_PATH="${OPENBLAS_PREFIX}/lib/pkgconfig:${LAPACK_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"

phpize
./configure
make -j$(sysctl -n hw.ncpu)
sudo make install
```

**3. Turn the extension on**

Open the `php.ini` shown by `php --ini` and add the line `extension=ndarray.so`. On Homebrew that file usually lives at `$(brew --prefix)/etc/php/<version>/php.ini`.

### Windows

**1. Get the tools you'll need**

- **Visual Studio 2019 or 2022** (Community is fine), or the standalone **Build Tools** package.
- **PHP SDK + development pack** for the PHP version you target — easiest is the [php-sdk-binary-tools](https://github.com/php/php-sdk-binary-tools) repository.
- **Prebuilt OpenBLAS for Windows x64** — download from the [OpenBLAS releases](https://github.com/OpenMathLib/OpenBLAS/releases) (the file looks like `OpenBLAS-0.3.30-x64.zip`). Unzip it anywhere.
- **(GPU only)** NVIDIA **CUDA Toolkit 12.x** for Windows x64. The installer sets the `%CUDA_PATH%` environment variable automatically.

**2. Build — CPU only**

Open the *x64 Native Tools Command Prompt* (or run `phpsdk-vs17-x64.bat` from the PHP SDK):

```bat
git clone https://github.com/RubixML/numpower.git
cd numpower

phpize
configure --enable-ndarray ^
          --with-openblas-dir=C:\path\to\openblas ^
          --with-prefix=C:\path\to\php
nmake
nmake install
```

After install, copy `libopenblas.dll` from the OpenBLAS `bin\` folder next to your `php.exe` (or anywhere on `%PATH%`). Then add `extension=php_ndarray.dll` to `php.ini`.

**3. Build with GPU (CUDA) support — optional**

The CUDA build needs one extra step compared to Linux. NumPower's GPU kernels live in `.cu` files, and the Windows build system (`nmake`) doesn't know how to compile those — so we hand them off to `nvcc.exe` first with a small script, then let `nmake` link the result in.

```bat
phpize
configure --enable-ndarray ^
          --with-openblas-dir=C:\path\to\openblas ^
          --with-cuda --with-cuda-dir="%CUDA_PATH%" ^
          --with-prefix=C:\path\to\php

REM Compiles the .cu files into ndarray_cuda.lib. Uses %CUDA_PATH% and
REM %PHP_PREFIX%; pass --cuda-path=... / --php-prefix=... if needed.
build-cuda-windows.bat

nmake
nmake install
```

If cuDNN lives outside the CUDA Toolkit folder, add `--with-cudnn-dir=C:\path\to\cudnn` to the `configure` line. Without cuDNN you still get cuBLAS-backed GPU math, just no GPU convolutions.

After `nmake install`, copy the CUDA runtime DLLs next to `php.exe` so the extension can find them at runtime:

```bat
copy "%CUDA_PATH%\bin\cublas64_*.dll"   C:\path\to\php\
copy "%CUDA_PATH%\bin\cudart64_*.dll"   C:\path\to\php\
copy "%CUDA_PATH%\bin\cublasLt64_*.dll" C:\path\to\php\
copy "%CUDA_PATH%\bin\cudnn64_*.dll"    C:\path\to\php\
```

> You only re-run `build-cuda-windows.bat` if you edit one of the `.cu` files. For regular development on the C parts, plain `nmake` is enough.

> **Don't want to install the CUDA Toolkit on Windows?** Build NumPower inside **WSL2 (Ubuntu)** instead and follow the Linux instructions. With the [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/) driver from NVIDIA your WSL2 PHP can talk to the host GPU just fine.

---

## GPU support

Linux GPU build in one block:

```bash
phpize
./configure --with-cuda
sudo make install-cuda      # NOT `make install` — that one skips the .cu files
```

Make sure before you start:

- Your NVIDIA driver is recent enough for the CUDA Toolkit version you have.
- `nvcc --version` works in your shell.
- `ldconfig -p | grep cublas` shows the cuBLAS library.

Want cuDNN (for `Conv1D` / `Conv2D` on GPU)? Install it from NVIDIA's site before the configure step — it'll be picked up automatically.

For Windows GPU instructions, see [Build with GPU (CUDA) support — optional](#windows) above.

You can build NumPower with `--with-cuda` even on a machine without a physical GPU (for example inside a container) — the build will succeed, and the runtime will simply throw an exception the moment you call `->gpu()` on a system with no device.

---

## Verifying the installation

```bash
php -m | grep ndarray
php -r '$a = NumPower::ones([3, 3]); echo $a;'
```

The second command should print a 3×3 matrix of ones. If you built with GPU support, try:

```bash
php -r 'echo (int)(new NDArray([1.0]))->gpu()->isGPU();'   # prints 1 when a CUDA device is reachable
```

---

## Using the GPU from your code

If you have an NVIDIA card with CUDA, you can move any `NDArray` to GPU memory and run operations there:

```php
$x = NumPower::ones([10, 10]);
$y = NumPower::ones([10, 10]);

$xGpu = $x->gpu();   // Copy $x from RAM to VRAM
$yGpu = $y->gpu();   // Copy $y from RAM to VRAM

$r = NumPower::matmul($xGpu, $yGpu);   // Multiplied on the GPU
```

NumPower manages both CPU and GPU memory for you — when an `NDArray` goes out of scope, its memory (in RAM or VRAM) is freed automatically. You can also move arrays back to CPU memory:

```php
$xCpu = $x->cpu();
```

> **Heads up:** every operation works on arrays that live on the *same* device. Adding a CPU array to a GPU array throws an exception — copy one of them across with `->cpu()` or `->gpu()` first.
