$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

$MODEL_PATH = "C:\Users\Emanuel\.cache\huggingface\hub\models--Freepik--flux.1-lite-8B-alpha\snapshots\812d376439b6e37b0e6f6dd401b2a98b1effacdb"
$OUTDIR = "resultados_comparacao"
$LOG_FILE = "benchmark_log.txt"

$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
$env:PYTORCH_ROCM_ARCH = "gfx1100"

New-Item -ItemType Directory -Force -Path "$OUTDIR\flux\baseline" | Out-Null
New-Item -ItemType Directory -Force -Path "$OUTDIR\flux\dpcache" | Out-Null
New-Item -ItemType Directory -Force -Path "$OUTDIR\flux\taylorseer" | Out-Null
New-Item -ItemType Directory -Force -Path "$OUTDIR\flux\teacache" | Out-Null

Write-Host "============================================"
Write-Host "  COMPARACAO FLUX (local)"
Write-Host "  Modelo: $MODEL_PATH"
Write-Host "============================================"
Write-Host "`nQuantos prompts usar?"
Write-Host "  1 - Prompt unico (rapido)"
Write-Host "  10 - 10 prompts de canhao (completo)"
Write-Host "  OU digite o numero de prompts que deseja`n"
$num_prompts = Read-Host "Quantidade"
if ([string]::IsNullOrWhiteSpace($num_prompts) -or $num_prompts -eq "1") {
    $PROMPT_FILE = "prompt_unico.txt"
} elseif ($num_prompts -eq "10") {
    $PROMPT_FILE = "prompts_cannon_10.txt"
} else {
    $PROMPT_FILE = "prompts_cannon_10.txt"
    Write-Host "Usando prompts_cannon_10.txt ($num_prompts prompts serao processados)"
}
Write-Host "Arquivo de prompts: $PROMPT_FILE`n"
Write-Host "`nQuais metodos rodar? (separados por virgula)"
Write-Host "  1 - DPCache"
Write-Host "  2 - TeaCache"
Write-Host "  3 - TaylorSeer"
Write-Host "  4 - Baseline (sem cache)"
Write-Host "  Ex: 2,3 (padrao: 2,3)`n"
$input = Read-Host "Escolha"
if ([string]::IsNullOrWhiteSpace($input)) { $input = "2,3" }
$escolhas = $input -split "," | ForEach-Object { $_.Trim() }

"==========================================" >> $LOG_FILE
"Benchmark $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" >> $LOG_FILE
"Prompt: $(Get-Content $PROMPT_FILE -Raw -Encoding UTF8).Trim()" >> $LOG_FILE
"==========================================" >> $LOG_FILE

if ($escolhas -contains "1") {
    Write-Host "`n--- DPCache FLUX ---"
    $t = Measure-Command { python dpcache_flux_infer.py --model_path $MODEL_PATH --prompt_file $PROMPT_FILE --output_path "$OUTDIR\flux\dpcache" }
    $elapsed = "{0:hh\:mm\:ss}" -f $t
    Write-Host "DPCache: $elapsed"
    "DPCache: $elapsed" >> $LOG_FILE
}

if ($escolhas -contains "2") {
    Write-Host "`n--- TeaCache FLUX ---"
    $t = Measure-Command { python teacache_flux_wrapper.py --model_path $MODEL_PATH --prompt_file $PROMPT_FILE --output "$OUTDIR\flux\teacache\saida.png" }
    $elapsed = "{0:hh\:mm\:ss}" -f $t
    Write-Host "TeaCache: $elapsed"
    "TeaCache: $elapsed" >> $LOG_FILE
}

if ($escolhas -contains "3") {
    Write-Host "`n--- TaylorSeer FLUX ---"
    $t = Measure-Command { python taylorseer_flux_infer.py --model_path $MODEL_PATH --prompt_file $PROMPT_FILE --output "$OUTDIR\flux\taylorseer\saida.png" }
    $elapsed = "{0:hh\:mm\:ss}" -f $t
    Write-Host "TaylorSeer: $elapsed"
    "TaylorSeer: $elapsed" >> $LOG_FILE
}

if ($escolhas -contains "4") {
    Write-Host "`n--- Baseline FLUX (sem cache) ---"
    $t = Measure-Command { python dpcache_flux_infer.py --model_path $MODEL_PATH --prompt_file $PROMPT_FILE --no_cache --output_path "$OUTDIR\flux\baseline" --num_steps 50 }
    $elapsed = "{0:hh\:mm\:ss}" -f $t
    Write-Host "Baseline: $elapsed"
    "Baseline (sem cache): $elapsed" >> $LOG_FILE
}

"==========================================" >> $LOG_FILE
Write-Host "`n`n============================================"
Write-Host "  FINALIZADO!"
Write-Host "  Resultados em: $OUTDIR"
Write-Host "  Log salvo em: $LOG_FILE"
Write-Host "============================================"
