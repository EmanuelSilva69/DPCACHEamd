# DPCACHEamd

**DPCache on AMD — Testing the original DPCache acceleration method on AMD hardware**

Este repositório contém adaptações e testes do [DPCache](https://github.com/amusi/DPCache) (Denoising as Path Planning: Training-Free Acceleration of Diffusion Models) em GPUs AMD.

## Sobre

O DPCache original acelera modelos de difusão via caching baseado em predição, utilizando programação dinâmica com *Path-Aware Cost Tensor (PACT)* para otimizar o schedule de caching.

Este fork/teste tem como objetivo:

- Avaliar a compatibilidade e performance do DPCache em hardware AMD (GPUs com ROCm)
- Adaptar scripts e dependências para funcionar com PyTorch + ROCm
- Documentar resultados, limitações e ajustes necessários

## Hardware de Teste

- **GPU AMD**: Radeon RX 7000 series / Instinct
- **SO**: Windows / Linux com ROCm
- **Framework**: PyTorch com suporte ROCm

## Status

- [x] Adaptação dos scripts de inferência FLUX para AMD
- [x] Testes de calibração em AMD

## Como Usar

```bash
pip install -r requirements.txt
```

### Inferência FLUX

```bash
python dpcache_flux_infer.py --mode infer --k 13 --first_full_steps 3 --dataset drawbench --sample_size 100 --output_path "flux_output"
```

### Calibração

```bash
python dpcache_flux_infer.py --mode calibrate --cali_prefix "flux_calibration" --dataset drawbench --sample_size 10 --output_path "calibration_results"
```

## Resultados

Os testes realizados com o DPCache em GPU AMD (configuração ROCm) demonstraram uma aceleração significativa no pipeline de inferência do modelo FLUX, permitindo a execução em hardware de consumo com latência reduzida.

### Tabela de Desempenho (10 Imagens)

| Método | Tempo Total | Latência Média/Img | Aceleração (Speedup) |
| :--- | :--- | :--- | :--- |
| **Baseline (Sem Cache)** | ~345 min | ~34,5 min | 1.00x |
| **DPCache (K=13)** | ~102 min | ~10,2 min | **~3.4x** |

### Destaques dos Experimentos
* **Aceleração Sustentada:** Observou-se uma redução consistente de mais de 3x no tempo total de inferência ao aplicar o *Dynamic Programming* para planejamento de trajetória no FLUX.
* **Validação em Dataset Regional:** O método foi validado com sucesso em datasets customizados (Dataset DrawBench e Dataset de patrimônio histórico — "Canhões de São Luís"), mantendo a fidelidade visual e a preservação de texturas finas.
* **Compatibilidade ROCm:** O DPCache provou ser funcional em arquiteturas AMD através do suporte a PyTorch+ROCm, utilizando a matriz de custos (PACT) para otimizar os saltos de passos (skips) de forma eficiente.

> *Nota: Os resultados acima refletem o ambiente de hardware de uso pessoal, sendo limitados principalmente pelo I/O de decodificação VAE do sistema.*

## Créditos

Projeto original: [DPCache](https://github.com/amusi/DPCache) — aceito no **CVPR 2026**.
