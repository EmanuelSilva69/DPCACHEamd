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
- [ ] Testes de calibração em AMD
- [ ] Comparação de performance com NVIDIA (baseline)
- [ ] Suporte para Wan2.1 em AMD

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

*(Em breve)*

## Créditos

Projeto original: [DPCache](https://github.com/amusi/DPCache) — aceito no **CVPR 2026**.
