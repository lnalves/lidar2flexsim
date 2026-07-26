# LiDAR PointNet++ para Warehouse

Projeto de machine learning para segmentação semântica e detecção de objetos
em nuvens de pontos do Warehouse LiDAR Dataset, usando PointNet++ em PyTorch.

O modelo aprende seis classes por ponto:

- `background`
- `Box`
- `ELFplusplus`
- `CargoBike`
- `FTS`
- `ForkLift`

Depois da segmentação, os pontos previstos são agrupados por classe e
convertidos em caixas 3D orientadas. O pacote também inclui calibração,
checkpoints seguros, retomada de treinamento e benchmark reproduzível.

## Estrutura

```text
ml/
├── cli.py                    # comandos train, infer e benchmark
├── gui.py                    # validação e execução usada pela interface
├── models/pointnet2_seg.py   # arquitetura PointNet++
├── data/pointnet.py          # leitura, labels e dataset PyTorch
├── preprocessing.py          # voxel, piso e outliers em NumPy
├── training.py               # treino, validação e checkpoints
├── inference.py              # segmentação e caixas 3D
├── evaluation.py             # IoU 3D e métricas de detecção
├── benchmark.py              # splits e relatórios reproduzíveis
└── configs/pointnet2_seg.yaml
```

`app.py` fornece a interface gráfica sobre esses mesmos comandos.

## Instalação

Recomenda-se Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para desenvolvimento:

```bash
python -m pip install -r requirements-dev.txt
```

## Dataset

A raiz do Warehouse LiDAR Dataset deve conter:

```text
dados/warehouse/
├── bin/
│   ├── 000000.bin
│   └── ...
└── label/
    ├── 000000.txt
    └── ...
```

Cada registro `.bin` contém quatro `float32`: `x`, `y`, `z` e intensidade.
Cada linha de label segue:

```text
classe cx cy cz dx dy dz yaw
```

As dimensões são dadas em metros e `yaw` em radianos.

## Interface gráfica

```bash
python app.py
```

A interface abre em `http://localhost:8080`. O fluxo principal pede apenas:

1. a pasta do Warehouse Dataset;
2. um scan;
3. um checkpoint;
4. o clique em **Analisar scan**.

Os parâmetros técnicos ficam recolhidos em **Ajustes avançados**. Treinamento
por presets, progresso, cancelamento e benchmark continuam disponíveis em
**Ferramentas do modelo**, sem competir com a análise principal.

Para abrir em uma janela nativa, instale `pywebview` e execute:

```bash
python -m pip install "pywebview>=5.4,<7"
python app.py --native
```

No modo navegador, o dataset padrão `dados/warehouse` é detectado
automaticamente. Também é possível informar outro caminho local e validá-lo.

## Treinamento

```bash
python -m ml.cli train \
  --dataset dados/warehouse \
  --config ml/configs/pointnet2_seg.yaml \
  --output runs \
  --device cpu \
  --class-weights auto
```

Cada execução cria uma pasta própria com `config.json`, `metadata.json`,
`history.jsonl` e `checkpoints/`. `best.pt` representa a melhor validação e
`last.pt` o estado mais recente.

Para uma validação rápida do fluxo:

```bash
python -m ml.cli train \
  --dataset dados/warehouse \
  --config ml/configs/pointnet2_seg.yaml \
  --output runs \
  --run-name quick-12x1024 \
  --max-scans 12 \
  --input-points 1024 \
  --epochs 2 \
  --batch-size 2 \
  --class-weights auto \
  --device cpu
```

`--max-scans` seleciona amostras ao longo da sequência e o split permanece
temporal, evitando vazamento entre frames consecutivos.

Para retomar uma execução:

```bash
python -m ml.cli train \
  --dataset dados/warehouse \
  --config ml/configs/pointnet2_seg.yaml \
  --output runs \
  --resume runs/20260723_213000_pointnet2
```

## Inferência

```bash
python -m ml.cli infer \
  --scan dados/warehouse/bin/000000.bin \
  --checkpoint runs/.../checkpoints/best.pt \
  --device cpu \
  --score-threshold 0.50
```

Filtros pós-segmentação podem ser aplicados por arquivo JSON:

```json
{
  "min_score": 0.55,
  "min_points": 8,
  "max_iou": 0.85,
  "per_class_min_dimensions": {
    "ForkLift": [0.3, 0.3, 0.3]
  }
}
```

```bash
python -m ml.cli infer \
  --scan dados/warehouse/bin/000000.bin \
  --checkpoint runs/.../checkpoints/best.pt \
  --calibration calibration.json
```

## Benchmark

```bash
python -m ml.cli benchmark \
  --dataset dados/warehouse \
  --checkpoint runs/.../checkpoints/best.pt \
  --max-scans 12 \
  --output benchmark/warehouse-12
```

O benchmark grava:

- `manifest.json`: IDs congelados de treino, validação e teste;
- `benchmark.json`: métricas de segmentação, IoU de caixas, tempo, ambiente e
  dados do checkpoint.

Para avaliar um JSON de predições separadamente:

```bash
python -m ml.evaluation \
  --predicoes predictions.json \
  --labels dados/warehouse/label \
  --scan-id 000000 \
  --class-aware \
  --saida benchmark/metricas_warehouse.json
```

## Testes

```bash
python -m pytest -q
python -m compileall -q ml
```

Os checkpoints são carregados com validação de arquitetura e desserialização
segura. Não use pesos de origem desconhecida sem revisar sua procedência.
