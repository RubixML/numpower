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
REM Required env vars (or arguments):
REM   %CUDA_PATH%   CUDA Toolkit install root (the dir that contains bin\,
REM                 include\, lib\x64\). May also be passed as first arg.
REM   %PHP_PREFIX%  PHP install root with include\php\... headers. Normally
REM                 set by setup-php-sdk; otherwise pass --php-prefix=<path>.
REM
REM Optional:
REM   %CUDNN_DIR%   cuDNN install root if cuDNN headers/libs are not already
REM                 sitting in %CUDA_PATH%. Only consulted for the -I path.
REM ===========================================================================

setlocal enabledelayedexpansion

set "ARG_CUDA_PATH="
set "ARG_PHP_PREFIX="
set "ARG_CUDNN_DIR="

:parse_args
if "%~1"=="" goto args_done
set "_arg=%~1"
if /I "!_arg:~0,13!"=="--cuda-path=" (
    set "ARG_CUDA_PATH=!_arg:~13!"
) else if /I "!_arg:~0,14!"=="--php-prefix=" (
    set "ARG_PHP_PREFIX=!_arg:~14!"
) else if /I "!_arg:~0,13!"=="--cudnn-dir=" (
    set "ARG_CUDNN_DIR=!_arg:~13!"
) else if "!_arg:~0,2!"=="--" (
    echo ERROR: unknown option "!_arg!"
    goto usage
) else if "!ARG_CUDA_PATH!"=="" (
    set "ARG_CUDA_PATH=!_arg!"
)
shift
goto parse_args
:args_done

if not "%ARG_CUDA_PATH%"=="" set "CUDA_PATH=%ARG_CUDA_PATH%"
if not "%ARG_PHP_PREFIX%"=="" set "PHP_PREFIX=%ARG_PHP_PREFIX%"
if not "%ARG_CUDNN_DIR%"=="" set "CUDNN_DIR=%ARG_CUDNN_DIR%"

if "%CUDA_PATH%"=="" (
    echo ERROR: CUDA_PATH not set. Either set the env var or pass the CUDA
    echo        Toolkit path as the first argument / --cuda-path=...
    goto usage
)
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo ERROR: nvcc.exe not found at "%CUDA_PATH%\bin\nvcc.exe"
    echo        Check that %%CUDA_PATH%% points at the toolkit root.
    exit /b 1
)

if "%PHP_PREFIX%"=="" (
    echo ERROR: PHP_PREFIX not set. The setup-php-sdk action exposes this as
    echo        its `prefix` output; locally, point it at the PHP install
    echo        whose include\php\... headers match your build target.
    goto usage
)

REM ---- Locate PHP include tree --------------------------------------------
REM PHP SDK installs headers under <prefix>\include\php\{main,Zend,TSRM,ext}.
REM Some older layouts drop them at <prefix>\include directly; probe both.
set "PHP_INC_ROOT="
if exist "%PHP_PREFIX%\include\php\main\php.h" (
    set "PHP_INC_ROOT=%PHP_PREFIX%\include\php"
) else if exist "%PHP_PREFIX%\include\main\php.h" (
    set "PHP_INC_ROOT=%PHP_PREFIX%\include"
)
if "%PHP_INC_ROOT%"=="" (
    echo ERROR: cannot locate php.h under "%PHP_PREFIX%\include".
    echo        Looked for: include\php\main\php.h and include\main\php.h.
    exit /b 1
)

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

:usage
echo.
echo Usage:
echo   build-cuda-windows.bat [--cuda-path=^<path^>] [--php-prefix=^<path^>] [--cudnn-dir=^<path^>]
echo.
echo Or rely on the matching env vars %%CUDA_PATH%%, %%PHP_PREFIX%%, %%CUDNN_DIR%%.
exit /b 1
