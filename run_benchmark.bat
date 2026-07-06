@echo off
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
setlocal enabledelayedexpansion

:: ============================================================
::  Benchmark: DPCache vs TeaCache vs TaylorSeer
::  Model: FLUX.1-lite-8B-alpha (Freepik)
:: ============================================================

set MODEL_PATH=C:\Users\Emanuel\.cache\huggingface\hub\models--Freepik--flux.1-lite-8B-alpha\snapshots\812d376439b6e37b0e6f6dd401b2a98b1effacdb
set PROMPT_FILE=prompt_unico.txt
set RESULT_FILE=benchmark_results.txt

:: Configuracoes AMD ROCm
set HSA_OVERRIDE_GFX_VERSION=11.0.0
set PYTORCH_ROCM_ARCH=gfx1100

echo Iniciando Benchmark em %DATE% %TIME% > %RESULT_FILE%

:: ============================================================
::  1. DPCache
:: ============================================================
echo [1/3] Rodando DPCache...
call conda activate naruto_env
(echo DPCache Start: %TIME%) >> %RESULT_FILE%
python dpcache_flux_infer.py --model_path "%MODEL_PATH%" --prompt_file %PROMPT_FILE% --output_path resultados_comparacao\flux\dpcache >> %RESULT_FILE% 2>&1
if %ERRORLEVEL% neq 0 echo DPCache FALHOU! >> %RESULT_FILE%
(echo DPCache End: %TIME%) >> %RESULT_FILE%
taskkill /F /IM python.exe >nul 2>&1

:: ============================================================
::  2. TeaCache
:: ============================================================
echo [2/3] Rodando TeaCache...
call conda activate teacache_env
(echo TeaCache Start: %TIME%) >> %RESULT_FILE%
python teacache_flux_wrapper.py --model_path "%MODEL_PATH%" --prompt_file %PROMPT_FILE% --output resultados_comparacao\flux\teacache\saida.png >> %RESULT_FILE% 2>&1
if %ERRORLEVEL% neq 0 echo TeaCache FALHOU! >> %RESULT_FILE%
(echo TeaCache End: %TIME%) >> %RESULT_FILE%
taskkill /F /IM python.exe >nul 2>&1

:: ============================================================
::  3. TaylorSeer
:: ============================================================
echo [3/3] Rodando TaylorSeer...
call conda activate taylorseer_env
(echo TaylorSeer Start: %TIME%) >> %RESULT_FILE%
python taylorseer_flux_infer.py --model_path "%MODEL_PATH%" --prompt_file %PROMPT_FILE% --output resultados_comparacao\flux\taylorseer\saida.png >> %RESULT_FILE% 2>&1
if %ERRORLEVEL% neq 0 echo TaylorSeer FALHOU! >> %RESULT_FILE%
(echo TaylorSeer End: %TIME%) >> %RESULT_FILE%
taskkill /F /IM python.exe >nul 2>&1

echo. >> %RESULT_FILE%
echo Benchmark Concluido. Veja %RESULT_FILE% para os logs de tempo.
pause
