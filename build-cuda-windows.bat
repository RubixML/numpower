@echo off
REM ===========================================================================
REM build-cuda-windows.bat — compile the CUDA (.cu) sources on Windows.
REM
REM NumPower's MSVC build (config.w32 + nmake) compiles all .c sources with
REM cl.exe, but the GPU kernels in src\ndmath\cuda\*.cu need nvcc.exe. nmake
REM has no built-in rule for .cu, so this script does the .cu -> .obj -> .lib
REM step out-of-band and produces ndarray_cuda.lib at the repo root. When
REM configure was run with --with-cuda, config.w32 adds ndarray_cuda.lib to
REM LIBS_NDARRAY so the subsequent `nmake` picks it up at link time.
REM
REM Run order on Windows:
REM   1. phpize
REM   2. configure --enable-ndarray --with-openblas-dir=... ^
REM                --with-cuda --with-cuda-dir=%CUDA_PATH% ^
REM                --with-prefix=<php-prefix>
REM   3. build-cuda-windows.bat            <-- THIS SCRIPT
REM   4. nmake
REM
REM Configuration is read from environment variables only — cmd.exe treats
REM `=` as a token delimiter inside batch arguments, which makes a
REM `--cuda-path=<value>` style flag fragile (the `=` splits the arg in two
REM and the value gets mis-parsed). The standard NVIDIA installer already
REM exports CUDA_PATH; setup-php-sdk exposes PHP_PREFIX. For local manual
REM use, `set CUDA_PATH=...` and `set PHP_PREFIX=...` before calling this
REM script.
REM
REM Required env vars:
REM   %CUDA_PATH%   CUDA Toolkit install root (the dir that contains bin\,
REM                 include\, lib\x64\).
REM   %PHP_PREFIX%  PHP install root. setup-php-sdk sets this to the binary
REM                 install (e.g. <root>\php-bin); the dev pack with the
REM                 actual PHP / Zend headers lives in a sibling <root>\php-dev
REM                 directory. The PHP-include probe below tries that layout
REM                 first, then a few install-tree fallbacks.
REM
REM Optional:
REM   %PHP_DEV_DIR% Explicit dev-pack root if our probes can't find php.h.
REM                 Must contain include\main\php.h.
REM   %CUDNN_DIR%   cuDNN install root if cuDNN headers/libs are not already
REM                 sitting in %CUDA_PATH%. Only consulted for the -I path.
REM ===========================================================================

setlocal enabledelayedexpansion

if "%CUDA_PATH%"=="" (
    echo ERROR: CUDA_PATH environment variable is not set.
    echo        The NVIDIA CUDA Toolkit installer normally sets it; otherwise
    echo        run `set CUDA_PATH=^<path^>` before calling this script.
    exit /b 1
)
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo ERROR: nvcc.exe not found at "%CUDA_PATH%\bin\nvcc.exe"
    echo        Check that %%CUDA_PATH%% points at the toolkit root.
    exit /b 1
)

if "%PHP_PREFIX%"=="" (
    echo ERROR: PHP_PREFIX environment variable is not set.
    echo        The setup-php-sdk action exposes this as its `prefix` output;
    echo        locally, point it at the PHP install whose include\php\...
    echo        headers match your build target.
    exit /b 1
)

REM ---- Locate PHP include tree --------------------------------------------
REM The PHP development pack (with main\php.h, Zend\, TSRM\, ext\) is laid
REM out differently depending on how PHP was installed. Probe in order; use
REM the first candidate where main\php.h exists:
REM
REM   1. %PHP_DEV_DIR%\include              — explicit override.
REM   2. <parent of PHP_PREFIX>\php-dev\include  — setup-php-sdk's layout:
REM      it installs PHP into <root>\php-bin\ and unpacks the dev pack
REM      into a sibling <root>\php-dev\ directory.
REM   3. %CD%                                — post-phpize: phpize.bat on
REM      Windows can unpack dev headers directly into the extension source
REM      tree, so main\php.h ends up alongside the .c files.
REM   4. %PHP_PREFIX%\include\php           — Linux-style install layout.
REM   5. %PHP_PREFIX%\include               — alternate install layout.
set "PHP_INC_ROOT="

if not "%PHP_DEV_DIR%"=="" (
    if exist "%PHP_DEV_DIR%\include\main\php.h" set "PHP_INC_ROOT=%PHP_DEV_DIR%\include"
)

if "%PHP_INC_ROOT%"=="" (
    REM Resolve <parent of PHP_PREFIX>\php-dev to a normalized absolute path
    REM via the %%~fI for-modifier (drops the trailing ..\).
    for %%I in ("%PHP_PREFIX%\..\php-dev") do (
        if exist "%%~fI\include\main\php.h" set "PHP_INC_ROOT=%%~fI\include"
    )
)

if "%PHP_INC_ROOT%"=="" (
    if exist "%CD%\main\php.h" set "PHP_INC_ROOT=%CD%"
)

if "%PHP_INC_ROOT%"=="" (
    if exist "%PHP_PREFIX%\include\php\main\php.h" set "PHP_INC_ROOT=%PHP_PREFIX%\include\php"
)

if "%PHP_INC_ROOT%"=="" (
    if exist "%PHP_PREFIX%\include\main\php.h" set "PHP_INC_ROOT=%PHP_PREFIX%\include"
)

if "%PHP_INC_ROOT%"=="" (
    echo ERROR: cannot locate php.h. Probed these layouts ^(in order^):
    if not "%PHP_DEV_DIR%"=="" echo   - %PHP_DEV_DIR%\include\main\php.h
    echo   - ^<parent of PHP_PREFIX^>\php-dev\include\main\php.h   ^(setup-php-sdk^)
    echo   - %CD%\main\php.h                                       ^(post-phpize^)
    echo   - %PHP_PREFIX%\include\php\main\php.h
    echo   - %PHP_PREFIX%\include\main\php.h
    echo.
    echo PHP_PREFIX  = %PHP_PREFIX%
    echo PHP_DEV_DIR = %PHP_DEV_DIR%
    echo.
    echo Set %%PHP_DEV_DIR%% to the dev pack root if your layout differs.
    exit /b 1
)

echo Using PHP headers from: %PHP_INC_ROOT%

set "PHP_INC=-I"%PHP_INC_ROOT%" -I"%PHP_INC_ROOT%\main" -I"%PHP_INC_ROOT%\Zend" -I"%PHP_INC_ROOT%\TSRM" -I"%PHP_INC_ROOT%\ext""

REM ---- nvcc invocation -----------------------------------------------------
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"

REM /MD matches PHP's default CRT (multi-threaded DLL) — must agree with the
REM cl.exe build of the .c sources or the link step will fail with LNK2038.
REM /EHsc enables standard C++ exception semantics (needed by <vector> in
REM cuda_dnn.cu). ZEND_ENABLE_STATIC_TSRMLS_CACHE mirrors what config.w32
REM passes to the .c compile so the headers see the same world.
set "HOST_FLAGS=/MD /EHsc /DZEND_ENABLE_STATIC_TSRMLS_CACHE=1 /D_USE_MATH_DEFINES /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS"
set "NVCC_DEFS=-DHAVE_CUBLAS=1 -DHAVE_NDARRAY=1 -DCOMPILE_DL_NDARRAY=1 -DZEND_WIN32=1 -DPHP_WIN32=1"

REM Pass the optional cuDNN include dir if the user set one. CUDA Toolkit
REM ships its own cuDNN headers when cuDNN is installed system-wide, in
REM which case CUDNN_DIR can be left empty.
set "CUDNN_INC="
if not "%CUDNN_DIR%"=="" (
    if exist "%CUDNN_DIR%\include\cudnn.h" (
        set "CUDNN_INC=-I"%CUDNN_DIR%\include""
        set "NVCC_DEFS=%NVCC_DEFS% -DHAVE_CUDNN=1"
    )
)

set "PROJ_INC=-I. -Isrc -Isrc\ndmath -Isrc\ndmath\cuda"

set "BUILD_DIR=cuda-build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo === Compiling src\ndmath\cuda\cuda_math.cu ===
"%NVCC%" -c %NVCC_DEFS% %PHP_INC% %PROJ_INC% %CUDNN_INC% ^
    -Xcompiler "%HOST_FLAGS%" ^
    src\ndmath\cuda\cuda_math.cu -o %BUILD_DIR%\cuda_math.obj
if errorlevel 1 (
    echo ERROR: cuda_math.cu failed to compile.
    exit /b 1
)

echo === Compiling src\ndmath\cuda\cuda_dnn.cu ===
"%NVCC%" -c %NVCC_DEFS% %PHP_INC% %PROJ_INC% %CUDNN_INC% ^
    -Xcompiler "%HOST_FLAGS%" ^
    src\ndmath\cuda\cuda_dnn.cu -o %BUILD_DIR%\cuda_dnn.obj
if errorlevel 1 (
    echo ERROR: cuda_dnn.cu failed to compile.
    exit /b 1
)

echo === Bundling .obj files into ndarray_cuda.lib ===
lib.exe /NOLOGO /OUT:ndarray_cuda.lib %BUILD_DIR%\cuda_math.obj %BUILD_DIR%\cuda_dnn.obj
if errorlevel 1 (
    echo ERROR: lib.exe failed.
    exit /b 1
)

echo.
echo OK: produced ndarray_cuda.lib (%BUILD_DIR%\ holds the intermediate .obj files).
echo     Now run `nmake` to link the extension.
exit /b 0
