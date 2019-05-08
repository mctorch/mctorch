if "%DEBUG%" == "1" (
  set BUILD_TYPE=debug
) ELSE (
  set BUILD_TYPE=release
)

set PATH=C:\Program Files\CMake\bin;C:\Program Files\7-Zip;C:\ProgramData\chocolatey\bin;C:\Program Files\Git\cmd;C:\Program Files\Amazon\AWSCLI;%PATH%


set INSTALLER_DIR=%SCRIPT_HELPERS_DIR%\installation-helpers

call %INSTALLER_DIR%\install_mkl.bat
call %INSTALLER_DIR%\install_magma.bat
call %INSTALLER_DIR%\install_sccache.bat
call %INSTALLER_DIR%\install_miniconda3.bat


:: Install ninja
if "%REBUILD%"=="" ( pip install -q ninja )

git submodule sync --recursive
git submodule update --init --recursive

set PATH=%TMP_DIR_WIN%\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp;%PATH%
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
set CUDA_PATH_V9_0=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
set NVTOOLSEXT_PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt
set CUDNN_LIB_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64
set CUDA_TOOLKIT_ROOT_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
set CUDNN_ROOT_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0

:: Target only our CI GPU machine's CUDA arch to speed up the build
set TORCH_CUDA_ARCH_LIST=5.2

sccache --stop-server
sccache --start-server
sccache --zero-stats
set CC=sccache cl
set CXX=sccache cl

set CMAKE_GENERATOR=Ninja

if not "%USE_CUDA%"=="1" (
  if "%REBUILD%"=="" (
    set NO_CUDA=1
    python setup.py install
  )
  if errorlevel 1 exit /b 1
  if not errorlevel 0 exit /b 1
)

if not "%USE_CUDA%"=="0" (
  if "%REBUILD%"=="" (
    sccache --show-stats
    sccache --zero-stats
    rd /s /q %CONDA_PARENT_DIR%\Miniconda3\Lib\site-packages\torch
    for /f "delims=" %%i in ('where /R caffe2\proto *.py') do (
      IF NOT "%%i" == "%CD%\caffe2\proto\__init__.py" (
        del /S /Q %%i
      )
    )
    copy %TMP_DIR_WIN%\bin\sccache.exe %TMP_DIR_WIN%\bin\nvcc.exe
  )

  set CUDA_NVCC_EXECUTABLE=%TMP_DIR_WIN%\bin\nvcc

  if "%REBUILD%"=="" set NO_CUDA=0

  python setup.py install --cmake && sccache --show-stats && (
    if "%BUILD_ENVIRONMENT%"=="" (
      echo NOTE: To run `import torch`, please make sure to activate the conda environment by running `call %CONDA_PARENT_DIR%\Miniconda3\Scripts\activate.bat %CONDA_PARENT_DIR%\Miniconda3` in Command Prompt before running Git Bash.
    ) else (
      mv %CD%\build\bin\test_api.exe %CONDA_PARENT_DIR%\Miniconda3\Lib\site-packages\torch\lib
      7z a %TMP_DIR_WIN%\%IMAGE_COMMIT_TAG%.7z %CONDA_PARENT_DIR%\Miniconda3\Lib\site-packages\torch %CONDA_PARENT_DIR%\Miniconda3\Lib\site-packages\caffe2 && python %SCRIPT_HELPERS_DIR%\upload_image.py %TMP_DIR_WIN%\%IMAGE_COMMIT_TAG%.7z
    )
  )
)

