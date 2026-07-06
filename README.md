# DPCACHEamd

**DPCache em AMD — Testes e adaptação do DPCache (Denoising as Path Planning) em hardware AMD com ROCm**

Este repositório contém uma suíte completa de testes, adaptações e comparações entre métodos de aceleração *training-free* para modelos de difusão (FLUX.1 e Wan2.1) rodando em **GPUs AMD** com PyTorch + ROCm.

## O que é o DPCache?

O [DPCache](https://github.com/amusi/DPCache) (aceito no CVPR 2026) é um método de aceleração *training-free* para modelos de difusão. Ele formula o processo de denoising como um problema de **planejamento de caminho (path planning)**, onde um *Cost Tensor 3D* (PACT — Path-Aware Cost Tensor) guia dinamicamente quais passos de inferência podem ser pulados via aproximação de Taylor, reduzindo o número de forward passes sem perda significativa de qualidade.

## O que foi feito aqui

Este fork/teste vai além da simples adaptação para AMD:

| Feature | Descrição |
|---|---|
| **DPCache (Taylor-DP)** | Implementação completa para FLUX (`dpcache_flux.py`) e Wan2.1 (`dpcache_wan.py`) com calibração + inferência |
| **TeaCache** | Wrapper de integração do [TeaCache](https://github.com/ali-vilab/TeaCache) para FLUX (`teacache_flux_wrapper.py`) + código original em `ModelosExtra/TeaCache/` |
| **TaylorSeer** | Integração do [TaylorSeer](https://github.com/Shen-Chenhui/TaylorSeer) para FLUX via diffusers (`taylorseer_flux_infer.py`) + código original em `ModelosExtra/TaylorSeer/` |
| **Wan2.1 I2V** | Pipeline Image-to-Video customizado (`wan_pipeline.py`) para o modelo Wan2.1-14B com suporte a DPCache |
| **Benchmark interativo** | `comparar_tudo.ps1` — script PowerShell que pergunta quais prompts usar (1 ou 10) e quais métodos rodar |
| **Benchmark automático** | `run_benchmark.bat` — executa DPCache, TeaCache e TaylorSeer em sequência com conda envs separados |
| **Avaliação de qualidade** | `avaliartudo.py` — compara todas as saídas contra baseline usando PSNR, SSIM, LPIPS e CLIP Score |
| **Suporte AMD ROCm** | Todas as variáveis de ambiente (`HSA_OVERRIDE_GFX_VERSION`, `PYTORCH_ROCM_ARCH`, etc.) já configuradas |

## Hardware de Teste

- **GPU:** AMD Radeon RX 7000 series (gfx1100)
- **SO:** Windows 11
- **Framework:** PyTorch 2.9.1 + ROCm
- **Modelos:** Freepik FLUX.1-lite-8B-alpha, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers

## Estrutura do Repositório

Status:
- [x] Adaptação dos scripts de inferência FLUX para AMD
- [x] Testes de calibração em AMD
- [x] Integração TeaCache + TaylorSeer
- [x] Benchmark interativo e automático
- [x] Suporte Wan2.1 I2V

```
DPCache/
├── dpcache/                    # Núcleo do DPCache (CacheHelper, calibração, schedule)
│   ├── __init__.py
│   ├── cache_utils.py          # CacheHelper, init_cache, gerenciamento de cache
│   ├── cali_utils.py           # Cálculo da cost matrix 3D, merge de resultados
│   └── schedule_utils.py       # Dynamic K-Selection, aproximação de Taylor, schedule
├── dpcache_flux.py             # Aplicação do DPCache ao FLUX (forwards substituídos)
├── dpcache_flux_infer.py       # Script de inferência/calibração FLUX com CLI
├── dpcache_wan.py              # Aplicação do DPCache ao Wan2.1
├── dpcache_wan_infer.py        # Script de inferência/calibração Wan2.1 I2V com CLI
├── teacache_flux_wrapper.py    # Wrapper do TeaCache para FLUX via Monkey Patch
├── taylorseer_flux_infer.py    # Integração do TaylorSeer para FLUX via diffusers
├── wan_pipeline.py             # Pipeline customizado WanI2VPipeline (diffusers)
├── avaliartudo.py              # Avaliação: PSNR, SSIM, LPIPS, CLIP Score
├── comparar_tudo.ps1           # Benchmark interativo (PowerShell)
├── run_benchmark.bat           # Benchmark automático (3 métodos, conda envs)
├── prompt_unico.txt            # Prompt único para testes rápidos
├── prompts_cannon_10.txt       # 10 prompts de canhão para testes completos
├── requirements.txt            # Dependências
├── ModelosExtra/
│   ├── TeaCache/               # Código original do TeaCache (FLUX, Wan2.1, etc.)
│   └── TaylorSeer/             # Código original do TaylorSeer (FLUX, Wan2.1, etc.)
├── datasets/                   # Datasets de prompts (DrawBench, PartiPrompts, canhoes)
├── canhoes_set/                # Imagens de entrada para testes com canhões
├── Imagembaseline/             # Imagens geradas sem cache (baseline)
├── imagemcachegato/            # Imagens geradas com DPCache (gato)
├── flux_test/                  # Imagens geradas com DPCache (canhão, 10 prompts)
├── resultados_comparacao/      # Resultados organizados por método
│   └── flux/
│       ├── baseline/           # Baseline (sem cache)
│       ├── dpcache/            # DPCache Taylor-DP
│       ├── taylorseer/         # TaylorSeer
│       └── teacache/           # TeaCache
├── cost_matrix_*.pkl           # Cost matrices calibradas
├── final_3d_cost_matrix_*.pkl  # Cost matrices finais (merge)
├── benchmark_log.txt           # Log do benchmark interativo
├── benchmark_results.txt       # Resultados do benchmark automático
├── log_gatos_dpcache.txt       # Log de teste com prompts de gato (DPCache)
├── log_gatos_baseline.txt      # Log de teste com prompts de gato (baseline)
└── log_canhoes_dpcache.txt     # Log de teste com prompts de canhão (DPCache)
```

## Métodos de Aceleração

### 1. DPCache (Taylor-DP) — `dpcache/`

Implementação do método Taylor-DP do DPCache:
1. **Calibração**: Executa todas as etapas completas, coletando o histórico de features em cada camada. No final, calcula um Cost Tensor 3D que mapeia o custo de pular de um passo para outro.
2. **Dynamic K-Selection**: Usa o cost tensor + programação dinâmica para selecionar os K passos ótimos que minimizam o erro acumulado.
3. **Inferência acelerada**: Nos passos não selecionados, usa aproximação de Taylor (derivadas de ordem 1 e 2) para estimar a saída sem executar os transformers blocks.

**Arquivos**: `dpcache/`, `dpcache_flux.py`, `dpcache_wan.py`

### 2. TeaCache — `teacache_flux_wrapper.py`

O [TeaCache](https://github.com/ali-vilab/TeaCache) usa um limiar de distância L1 relativa acumulada para decidir quando pular etapas. O wrapper substitui o `forward()` do `FluxTransformer2DModel` por uma versão que monitora a diferença entre entradas moduladas e pula blocos quando a diferença acumulada fica abaixo do limiar (`rel_l1_thresh`, default 0.6).

**Arquivo**: `teacache_flux_wrapper.py` + `ModelosExtra/TeaCache/TeaCache4FLUX/`

### 3. TaylorSeer — `taylorseer_flux_infer.py`

O [TaylorSeer](https://github.com/Shen-Chenhui/TaylorSeer) também usa expansão de Taylor, mas implementa o cache em nível de bloco com `cache_functions.py` próprio. A integração carrega os forwards customizados de `ModelosExtra/TaylorSeer/TaylorSeers-Diffusers/taylorseer_flux/`.

**Arquivo**: `taylorseer_flux_infer.py` + `ModelosExtra/TaylorSeer/TaylorSeers-Diffusers/`

## Scripts de Benchmark

### `comparar_tudo.ps1` (Interativo — PowerShell)

Script interativo que:
1. Pergunta **quantos prompts usar** (1 = rápido, 10 = completo, ou qualquer número)
2. Carrega `prompt_unico.txt` (1 prompt) ou `prompts_cannon_10.txt` (10 prompts)
3. Pergunta **quais métodos rodar** (1=DPCache, 2=TeaCache, 3=TaylorSeer, 4=Baseline)
4. Executa cada método selecionado, medindo o tempo com `Measure-Command`
5. Salva tudo em `resultados_comparacao/flux/<metodo>/` e registra no `benchmark_log.txt`

```powershell
.\comparar_tudo.ps1
```

### `run_benchmark.bat` (Automático — CMD)

Benchmark sequencial que:
1. Ativa `conda activate naruto_env` para DPCache
2. Ativa `conda activate teacache_env` para TeaCache
3. Ativa `conda activate taylorseer_env` para TaylorSeer
4. Mata processos Python entre execuções com `taskkill`
5. Salva timestamps em `benchmark_results.txt`

```batch
run_benchmark.bat
```

### Prompts

| Arquivo | Qtd | Descrição |
|---|---|---|
| `prompt_unico.txt` | 1 | Close-up de um canhão histórico em São Luís, Maranhão |
| `prompts_cannon_10.txt` | 10 | Variações do mesmo tema: diferentes ângulos, iluminação, composição |

## Suporte a Wan2.1 (Image-to-Video)

O arquivo `wan_pipeline.py` implementa um pipeline customizado `WanI2VPipeline` usando os componentes do `diffusers`. O `dpcache_wan_infer.py` permite:

- **Calibração**: `python dpcache_wan_infer.py --mode calibrate --dataset_dir datasets/ --sample_size 10`
- **Inferência**: `python dpcache_wan_infer.py --mode infer --image_path test.jpg --prompt "a blue car" --k 12`
- **Sem cache**: `python dpcache_wan_infer.py --mode infer --no_cache`

O DPCache é aplicado via monkey patch no `WanTransformer3DModel`, substituindo o `forward()` do transformer e de cada bloco. A calibração é feita na `cond_stream`, resultando no cost matrix `final_3d_cost_matrix_wan_cfg_3.pkl`.

## Avaliação de Qualidade

O `avaliartudo.py` compara as imagens geradas por cada método acelerado contra a baseline (sem cache):

| Métrica | Descrição |
|---|---|
| **PSNR** | Peak Signal-to-Noise Ratio (maior = melhor) |
| **SSIM** | Structural Similarity (maior = melhor) |
| **LPIPS** | Learned Perceptual Image Patch Similarity (menor = melhor) |
| **CLIP Score** | Similaridade textual-visual (maior = melhor) |

```bash
python avaliartudo.py
```

## Passo a Passo Completo (Reprodutibilidade)

### 1. Clonar e instalar

```bash
git clone https://github.com/EmanuelSilva69/DPCACHEamd.git
cd DPCACHEamd
pip install -r requirements.txt
```

### 2. Baixar o modelo FLUX

O modelo usado é o [Freepik/flux.1-lite-8B-alpha](https://huggingface.co/Freepik/flux.1-lite-8B-alpha):

```python
from huggingface_hub import snapshot_download
snapshot_download("Freepik/flux.1-lite-8B-alpha", local_dir="caminho/para/modelo")
```

Atualize o `MODEL_PATH_DEFAULT` nos scripts com o caminho local.

### 3. Calibrar o DPCache (gerar cost matrix)

```bash
python dpcache_flux_infer.py --mode calibrate --cali_prefix "flux_calibration" --dataset drawbench --sample_size 10
```

Isso gera `cost_matrix_flux_calibration_2.pkl` e faz o merge para `final_cost_matrix_flux_calibration.pkl`.

### 4. Inferência com DPCache

```bash
python dpcache_flux_infer.py --mode infer --k 13 --first_full_steps 3 --cost_matrix_path "final_3d_cost_matrix_flux.pkl" --prompt_file "prompt_unico.txt" --output_path "resultados_comparacao/flux/dpcache"
```

### 5. Inferência com TeaCache

```bash
python teacache_flux_wrapper.py --prompt_file "prompt_unico.txt" --thresh 0.6 --steps 28 --output "resultados_comparacao/flux/teacache/saida.png"
```

### 6. Inferência com TaylorSeer

```bash
python taylorseer_flux_infer.py --prompt_file "prompt_unico.txt" --steps 50 --output "resultados_comparacao/flux/taylorseer/saida.png"
```

### 7. Baseline (sem cache)

```bash
python dpcache_flux_infer.py --mode infer --no_cache --num_steps 50 --prompt_file "prompt_unico.txt" --output_path "resultados_comparacao/flux/baseline"
```

### 8. Benchmark completo (tudo de uma vez)

```powershell
.\comparar_tudo.ps1
```

### 9. Avaliar resultados

```bash
python avaliartudo.py
```

### 10. Wan2.1 — Calibração

```bash
python dpcache_wan_infer.py --mode calibrate --dataset_dir "caminho/dataset" --sample_size 10 --cali_prefix "wan_cfg_3_720p"
```

### 11. Wan2.1 — Inferência com DPCache

```bash
python dpcache_wan_infer.py --mode infer --image_path "test.jpg" --prompt "a blue car driving down a dirt road" --k 12 --cost_matrix_path "final_3d_cost_matrix_wan_cfg_3.pkl"
```

## Resultados Iniciais

### Benchmark Automático (28/06/2026)

O `benchmark_results.txt` registra a execução dos 3 métodos no hardware AMD. O DPCache foi executado com schedule de 13 passos selecionados (k=13) em 50 passos totais. O erro `torch.distributed.is_initialized` foi encontrado durante a execução inicial, indicando necessidade de ajuste na versão do PyTorch/distributed.

### Cost Matrices Geradas

| Arquivo | Modelo | Descrição |
|---|---|---|
| `final_3d_cost_matrix_flux.pkl` | FLUX | Cost matrix final (3D) para FLUX |
| `final_3d_cost_matrix_wan_cfg_3.pkl` | Wan2.1 | Cost matrix final para Wan2.1 com CFG=3 |
| `cost_matrix_canhoes_set_2.pkl` | FLUX | Cost matrix calibrada no dataset de canhões |
| `cost_matrix_flux_amd_2.pkl` | FLUX | Cost matrix calibrada no hardware AMD |

## Problemas Conhecidos

- **`torch.distributed.is_initialized`**: O PyTorch 2.9.1 moveu `is_initialized` para `torch.distributed._functional`. O `cache_utils.py` precisa de ajuste para compatibilidade. Solução temporária: modificar para `try: import torch.distributed as dist; dist.is_initialized()`.
- **ModelosExtra com `.git` interno**: Os diretórios `ModelosExtra/TeaCache` e `ModelosExtra/TaylorSeer` continham repositórios git independentes. Foram removidos para permitir versionamento direto.
- **Consumo de VRAM**: O DPCache reduz o número de passos, mas o pico de VRAM durante a calibração é maior (precisa armazenar histórico de features).

### Tabela de Desempenho (1 prompt — modelo FLUX.1-lite-8B-alpha, GPU AMD RX 7000)

| Método | Tempo | Speedup vs Baseline | Observação |
| :--- | :--- | :--- | :--- |
| **Baseline (sem cache)** | 51min 09s | 1.00x | Inferência completa, 50 steps |
| **TeaCache** (thresh=0.6) | 08min 45s | **~5.8x** | 28 steps, aceleração por diferença L1 |
| **TaylorSeer** | 15min 00s | **~3.4x** | 50 steps com aproximação de Taylor |
| **DPCache** (k=13) | 12min 35s  | **4,06x**| 43 steps |

### Destaques dos Experimentos
* **Aceleração Sustentada:** Observou-se uma redução consistente de mais de 3x no tempo total de inferência ao aplicar o *Dynamic Programming* para planejamento de trajetória no FLUX.
* **Validação em Dataset Regional:** O método foi validado com sucesso em datasets customizados (Dataset DrawBench e Dataset de patrimônio histórico — "Canhões de São Luís"), mantendo a fidelidade visual e a preservação de texturas finas.
* **Compatibilidade ROCm:** O DPCache provou ser funcional em arquiteturas AMD através do suporte a PyTorch+ROCm, utilizando a matriz de custos (PACT) para otimizar os saltos de passos (skips) de forma eficiente.

> *Nota: Os resultados acima refletem o ambiente de hardware de uso pessoal, sendo limitados principalmente pelo I/O de decodificação VAE do sistema.*

## Créditos

- **DPCache**: [DPCache](https://github.com/amusi/DPCache) — aceito no CVPR 2026
- **TeaCache**: [TeaCache](https://github.com/ali-vilab/TeaCache)
- **TaylorSeer**: [TaylorSeer](https://github.com/Shen-Chenhui/TaylorSeer)
- **FLUX**: [Freepik/flux.1-lite-8B-alpha](https://huggingface.co/Freepik/flux.1-lite-8B-alpha)
- **Wan2.1**: [Wan-AI/Wan2.1](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers)

---

**Licença**: Apache 2.0

---

*Documentação gerada em 06/07/2026 — todo o código, logs e resultados estão versionados neste repositório.*
