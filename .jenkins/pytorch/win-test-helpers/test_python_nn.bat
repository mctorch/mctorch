call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat


pushd test

echo Some smoke tests
python %SCRIPT_HELPERS_DIR%\run_python_nn_smoketests.py
if ERRORLEVEL 1 exit /b 1

echo Run nn tests
python run_test.py --include nn --verbose
if ERRORLEVEL 1 exit /b 1

popd


