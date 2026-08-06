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
├── cli.py                    # comandos train, infer, benchmark e stream
├── gui.py                    # validação e execução usada pela interface
├── models/pointnet2_seg.py   # arquitetura PointNet++
├── data/pointnet.py          # leitura, labels e dataset PyTorch
├── preprocessing.py          # voxel, piso e outliers em NumPy
├── training.py               # treino, validação e checkpoints
├── inference.py              # segmentação e caixas 3D
├── evaluation.py             # IoU 3D e métricas de detecção
├── benchmark.py              # splits e relatórios reproduzíveis
├── flexsim/                  # ponte em tempo real com o FlexSim
│   ├── sources.py            # fontes de quadros: replay e ao vivo
│   ├── pipeline.py           # laço persistente fonte → modelo → cena
│   ├── tracking.py           # IDs persistentes entre quadros
│   ├── transform.py          # sensor → referencial do modelo FlexSim
│   ├── scene.py              # contrato de cena (JSON e CSV)
│   ├── export.py             # arquivos atômicos, FlexScript e STL
│   └── server.py             # servidor HTTP da cena atual
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

O split é temporal e tem três blocos: treino, validação e um bloco final de
teste (`--test-fraction`, padrão 0,1) que nenhum loader consome. Os três ficam
registrados em `metadata.json`, e é isso que permite ao benchmark reproduzir
exatamente o que o checkpoint viu.

Para uma validação rápida do fluxo:

```bash
python -m ml.cli train \
  --dataset dados/warehouse \
  --config ml/configs/pointnet2_seg.yaml \
  --output runs \
  --run-name quick-12 \
  --max-scans 12 \
  --epochs 2 \
  --batch-size 2 \
  --class-weights auto \
  --device cpu
```

`--max-scans` amostra a sequência com espaçamento uniforme, e não os primeiros
scans: um recorte contíguo cobre poucos segundos de gravação e deixa a maioria
das classes de fora.

O padrão de `input_points` é 8192. Os scans do Warehouse têm entre 3,5k e 9k
pontos, com menos de 7% deles dentro de caixas; amostrar menos que isso descarta
justamente os pontos de objeto que o segmentador precisa aprender.

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
  --checkpoint runs/exemplo/checkpoints/best.pt \
  --from-run runs/exemplo \
  --output benchmark/exemplo
```

`--from-run` reaproveita o split gravado pela execução. Sem ele o benchmark
monta o próprio recorte do dataset, e aí os blocos rotulados como treino e
validação não são os que o checkpoint realmente usou. A interface passa a flag
sozinha quando o checkpoint escolhido pertence a uma execução.

O benchmark grava:

- `manifest.json`: IDs congelados de treino, validação e teste;
- `benchmark.json`: métricas de segmentação, IoU de caixas, tempo, ambiente e
  dados do checkpoint.

O formato de predição (`classe`, `class_id`, `score`, `centro`, `dimensoes`,
`rotacao`, `num_pontos`) é o contrato de entrada da ponte FlexSim e não deve
mudar sem necessidade. O bloco de teste reservado pelo treino é o conjunto
natural para validar a ponte.

Para avaliar um JSON de predições separadamente:

```bash
python -m ml.evaluation \
  --predicoes predictions.json \
  --labels dados/warehouse/label \
  --scan-id 000000 \
  --class-aware \
  --saida benchmark/metricas_warehouse.json
```

## Integração com o FlexSim em tempo real

O comando `stream` mantém um processo vivo: carrega o checkpoint uma única
vez, consome quadros de uma fonte, segmenta, rastreia os objetos entre
quadros e publica a cena resultante.

```bash
python -m ml.cli stream \
  --dataset dados/warehouse \
  --checkpoint runs/exemplo/checkpoints/best.pt \
  --output-dir flexsim \
  --serve \
  --rate 10
```

Enquanto não há sensor, a fonte é um replay do dataset no ritmo de um LiDAR
real (`--rate`, em Hz). O laço emite um JSON por quadro, com o orçamento de
tempo medido, e um resumo ao final:

```json
{"event": "scene", "frame": 19, "objects": 22, "detections": 11,
 "inference_ms": 14.44, "total_ms": 17.21, "fps": 58.1}
```

### Como o FlexSim recebe a cena

Os dois caminhos publicam exatamente o mesmo conteúdo e podem coexistir:

- **Arquivo** (`--output-dir`): grava `scene.json` e `scene.csv` por rename
  atômico, então o FlexSim nunca lê um arquivo pela metade. Funciona sem
  rede, inclusive por pasta compartilhada.
- **HTTP** (`--serve`): serve `GET /scene.json`, `GET /scene.csv` e
  `GET /health` em `127.0.0.1:8765`. Use `--serve-host 0.0.0.0` apenas se o
  FlexSim rodar em outra máquina — o padrão não expõe nada à rede.

O `--output-dir` também recebe `lidar_bridge.txt`, o FlexScript a colar em
**Tools > User Commands**. Ele é idempotente: cria o que falta, atualiza pelo
nome o que já existe e destrói só o que a cena marcou como `removed`. É essa
diferença que preserva estatísticas e tarefas em andamento na simulação.

Instalação no modelo, uma vez:

1. crie um container vazio chamado `LidarScene` (ajustável com `--container`);
2. cole `lidar_bridge.txt` como User Command `atualizarDoLidar`;
3. chame `atualizarDoLidar()` num Process Flow em laço, a cada 0,1 s.

O script tem duas linhas marcadas com `AJUSTE`, isoladas no topo: a criação
de objeto e a divisão de campos, que variam entre versões do FlexSim.

O CSV existe porque o FlexScript lê arquivos linha a linha em qualquer
versão, mas não tem parser de JSON garantido. A primeira linha é
`formato,frame,timestamp,linhas`, a segunda são os nomes das colunas.

### Calibração e mapeamento

O detector trabalha no referencial do sensor, em metros e radianos. O FlexSim
usa graus e a origem do próprio modelo, e ancora objetos pelo **canto** da
bounding box — não pelo centro. A cena publica `center` e `location` (canto)
lado a lado, e `anchor: "corner"` declara qual deles o FlexScript usa.

```bash
python -m ml.cli stream ... \
  --sensor-translation 12.5,4.0,0.0 \
  --sensor-yaw 90
```

O mapa de classe para tipo FlexSim tem um padrão (`Box` → `VisualTool`, as
quatro classes móveis → `Transporter`) e é substituível por JSON:

```json
{
  "flexsim_objects": {
    "Box": "Rack",
    "ForkLift": "Transporter"
  }
}
```

```bash
python -m ml.cli stream ... --flexsim-map mapa.json
```

### Tracking

Sem identidade temporal, cada atualização seria "apague tudo e recrie". O
tracker associa detecções a faixas persistentes por proximidade no plano
`xy`, suaviza a geometria e estima velocidade em m/s. Uma faixa só é
publicada depois de `--track-min-hits` observações e sobrevive
`--track-max-age` quadros sem ser vista, o que evita que uma oclusão breve
destrua um objeto no FlexSim.

Os padrões (gate de 1,5 m, `max_age=5`, `min_hits=2`) foram escolhidos para
10 Hz em armazém. Com um detector instável eles produzem mais objetos
publicados do que detecções por quadro, porque as faixas seguem em
`coasting`; vale reapertá-los junto com a melhoria do checkpoint.

### Quando o sensor chegar

Toda a ponte é escrita contra `PointSource`, que entrega quadros `[N, 4]`.
O driver do sensor entra como uma implementação nova, sem tocar em tracking,
exportador ou servidor:

```python
from ml.flexsim import LivePointSource

source = LivePointSource()
# no receptor do sensor, a cada rotação completa:
source.push(pontos_xyzi)
```

`LivePointSource` guarda apenas o quadro mais recente e conta os descartes:
acumular fila só aumentaria a defasagem entre o armazém real e a simulação.

## Testes

```bash
python -m pytest -q
python -m compileall -q ml
```

Os checkpoints são carregados com validação de arquitetura e desserialização
segura. Não use pesos de origem desconhecida sem revisar sua procedência.
