# lidar2flexsim — Protótipo LiDAR → FlexSim

Protótipo que identifica objetos industriais (esteiras, bancadas, operadores) em nuvens de pontos LiDAR e gera arquivos importáveis no FlexSim. Desenvolvido como base para o plano de trabalho PUIC 2026 (gêmeos digitais com dados visuais).

## Como funciona

O pipeline segue cinco etapas clássicas de percepção 3D:

1. **Pré-processamento** — downsampling por voxel (reduz custo computacional preservando a forma) e remoção estatística de outliers (elimina ruído do sensor). O pipeline aceita `.bin` do Warehouse LiDAR (`x, y, z, intensidade`) diretamente, além de `.ply`, `.pcd` e `.xyz`.
2. **Remoção do chão** — RANSAC procura um plano entre os pontos da faixa inferior de `z` e rejeita planos cuja normal seja muito inclinada. Isso evita remover uma parede quando ela tem mais retornos que o piso.
3. **Clusterização** — DBSCAN agrupa pontos em objetos individuais. Em nuvens esparsas do VLP-16, o modo `BEV` projeta os pontos no plano XY antes de clusterizar e reduz a fragmentação entre partes do mesmo objeto.
4. **Classificação heurística** — regras geométricas sobre a bounding box de cada cluster: operadores são altos (1,3–2,1 m) e estreitos; esteiras são compridas (razão comprimento/largura ≥ 2,5); bancadas têm altura de mesa (0,6–1,3 m).
5. **Exportação** — três formatos para o FlexSim:
   - `modelos_3d/*.stl` — uma malha por objeto (importável em Visuals > Shape);
   - `layout.json` / `layout.csv` — posição, dimensões, rotação e classe de cada objeto (útil para auto-build orientado a dados);
   - `build_flexsim.txt` — FlexScript que recria o layout com objetos da biblioteca padrão (Conveyor, Processor, Operator). Cole no Script Console do FlexSim. O mapeamento pode ser sobrescrito por um JSON.

## Uso

```bash
pip install open3d numpy

# gerar cena sintética de teste (chão + esteira + bancada + 2 operadores)
python gerar_cena_demo.py

# executar o pipeline
python lidar2flexsim.py cena_demo.ply --saida ./saida --eps 0.22 --min-points 60

# gerar imagem da segmentação
python visualizar.py
```

### Interface gráfica

Para usar o pipeline sem digitar comandos, existe uma interface desktop em
`app.py`. Ela seleciona a pasta local do dataset, oferece predefinições de
parâmetros, executa scans em segundo plano, mostra o progresso e exporta os
arquivos para o FlexSim.

Recomenda-se Python 3.11 para manter compatibilidade com Open3D:

```bash
python3.11 -m venv .venv
source .venv/bin/activate             # macOS/Linux
python -m pip install -r requirements.txt
python app.py
```

Quando o backend nativo estiver disponível, o aplicativo abre em uma janela
local. Caso contrário, ele pode ser executado no navegador local. O botão
`Selecionar pasta` usa o seletor nativo quando possível; também é possível
digitar o caminho manualmente.

O fluxo da interface é:

1. selecionar a raiz do dataset (ou a pasta `bin/`);
2. validar `bin/`, `label/` e `vis/`;
3. escolher um scan, um intervalo ou todos os scans;
4. selecionar o perfil `Rápido`, `Equilibrado` ou `Detalhado`;
5. iniciar, acompanhar ou cancelar o processamento;
6. consultar resultados e métricas;
7. exportar `layout.json`, `layout.csv` e `build_flexsim.txt`.

Para preparar o empacotamento desktop, instale também o backend opcional:

```bash
python -m pip install -r requirements-desktop.txt
```

A interface chama diretamente os serviços Python em `core/`; a CLI permanece
disponível para experimentos reproduzíveis e automação.

Depois de validar a execução local, o pacote desktop pode ser gerado com:

```bash
python -m pip install -r requirements-desktop.txt
nicegui-pack --onefile --name "LiDAR2FlexSim" app.py
```

O artefato será colocado em `dist/`. O empacotamento deve ser feito no mesmo
sistema operacional em que o aplicativo será distribuído.

### Warehouse LiDAR Dataset

Baixe o conteúdo da [pasta oficial no Google Drive](https://drive.google.com/drive/folders/1T0hDyBnyY22pwShCDjSK95hzItTqqLqf) e mantenha a estrutura `bin/`, `label/` e `vis/`. O dataset tem 3.287 scans consecutivos e cinco classes: `Box`, `ELFplusplus`, `CargoBike`, `FTS` e `ForkLift`. Ele não possui anotações de pessoas.

Um scan pode ser processado diretamente:

```bash
python lidar2flexsim.py dados/warehouse/bin/000000.bin \
  --saida saida/000000 --voxel 0.05 --eps 0.25 --min-points 20 \
  --plane-distance 0.05 --oriented-box --cluster-mode bev \
  --config config/warehouse_classes.json
```

Para converter arquivos binários para inspeção em outras ferramentas:

```bash
python converter_warehouse_bin.py dados/warehouse/bin/000000.bin \
  --saida dados/pcd/000000.pcd
```

Para processar vários scans sem gerar STL de cada um:

```bash
python processar_warehouse.py dados/warehouse/bin \
  --saida saida/predicoes_warehouse.json \
  --voxel 0.05 --eps 0.25 --min-points 20 \
  --plane-distance 0.05 --oriented-box --cluster-mode bev
```

### Backend PointNet++ (opcional)

O projeto também inclui um segmentador supervisionado no estilo PointNet++.
Ele aprende características locais hierárquicas diretamente dos pontos e
substitui a classificação geométrica por ponto. As caixas orientadas do
Warehouse LiDAR Dataset são rasterizadas automaticamente em labels de seis
classes: `background`, `Box`, `ELFplusplus`, `CargoBike`, `FTS` e `ForkLift`.
O treinamento usa divisão temporal dos scans, evitando misturar frames
consecutivos entre treino e validação.

As dependências de PyTorch são opcionais e ficam separadas da instalação da
interface:

```bash
python -m pip install -r requirements-ml.txt
```

Treine um primeiro modelo em CPU (para uma avaliação real, aumente o número
de épocas e prefira uma GPU):

```bash
python -m ml.cli train \
  --dataset dados/warehouse \
  --config ml/configs/pointnet2_seg.yaml \
  --output checkpoints \
  --device cpu \
  --class-weights 0.1,1,1,1,1,1
```

`--class-weights` é opcional, mas costuma ajudar porque o fundo ocupa a maior
parte dos pontos; a ordem é `background`, `Box`, `ELFplusplus`, `CargoBike`,
`FTS`, `ForkLift`.

Cada época grava um checkpoint em `checkpoints/`. Para testar um scan:

```bash
python -m ml.cli infer \
  --scan dados/warehouse/bin/000000.bin \
  --checkpoint checkpoints/pointnet2_epoch_0020.pt \
  --device cpu \
  --score-threshold 0.50
```

O mesmo backend pode ser usado na CLI principal, já com exportação para o
FlexSim:

```bash
python lidar2flexsim.py dados/warehouse/bin/000000.bin \
  --backend pointnet2 \
  --checkpoint checkpoints/pointnet2_epoch_0020.pt \
  --saida saida/000000-pointnet2 \
  --config config/warehouse_classes.json
```

Na interface gráfica, selecione `PointNet++ (segmentação)`, informe o
checkpoint, escolha o dispositivo e ajuste o limiar de confiança. O serviço
gera as mesmas previsões `layout.json`, `layout.csv`, `build_flexsim.txt` e
STLs do backend heurístico; sem PyTorch ou sem checkpoint, o backend
heurístico continua sendo a opção padrão.

O modelo é uma implementação portátil em PyTorch das ideias de abstração de
conjuntos e propagação de características do PointNet++, sem extensões CUDA e
sem exigir PyTorch Geometric. O arquivo `ml/configs/pointnet2_seg.yaml`
concentra os hiperparâmetros para facilitar os experimentos.
Para a fundamentação teórica, consulte o artigo original
[PointNet++](https://arxiv.org/abs/1706.02413).

O arquivo de labels correspondente a `000000.bin` é `label/000000.txt`. Cada linha contém `classe x y z dimensao_x dimensao_y dimensao_z yaw`, com dimensões em metros e `yaw` em radianos.

Avalie as caixas previstas com IoU 3D. Por padrão, a avaliação é geométrica e
ignora a classe, o que também permite comparar a heurística legada com o
modelo. Para medir a ontologia aprendida pelo PointNet++, use `--class-aware`:

```bash
python avaliar_deteccoes.py \
  --predicoes saida/predicoes_warehouse.json \
  --labels dados/warehouse/label \
  --saida saida/metricas_warehouse.json
```

O avaliador calcula precisão, recall, F1, IoU médio, erro de centro, erro dimensional e erro de `yaw` para IoU 0,25 e 0,50. A opção `--class-aware` só deve ser usada depois que um classificador produzir as cinco classes do dataset; nesse caso, `--class-map` pode fornecer um mapeamento entre nomes.

Para um único `layout.json`:

```bash
python avaliar_deteccoes.py \
  --predicoes saida/000000/layout.json \
  --labels dados/warehouse/label/000000.txt
```

O repositório oficial e o artigo devem ser citados no trabalho. O dataset é distribuído sob CC-BY-SA-4.0.

## Parâmetros importantes

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--voxel` | 0.03 m | Resolução após downsampling. Menor = mais detalhe, mais lento. |
| `--eps` | 0.15 m | Raio do DBSCAN. Pequeno demais fragmenta objetos; grande demais funde objetos próximos. |
| `--min-points` | 40 | Filtra clusters pequenos (ruído/fragmentos). |
| `--plane-distance` | 0.03 m | Tolerância do RANSAC para o plano do piso. |
| `--max-ground-tilt` | 25° | Inclinação máxima da normal do piso em relação ao eixo `z`. Planos mais inclinados são rejeitados. |
| `--ground-quantile` | 0.30 | Fração inferior de pontos usada como candidata ao piso. |
| `--cluster-mode` | `3d` | Clusteriza em XYZ ou na projeção XY; `bev` é uma alternativa experimental para o VLP-16. |
| `--outlier-neighbors` | 12 | Vizinhos do filtro estatístico. Use `--no-outlier-filter` para preservar todos os retornos. |
| `--outlier-std-ratio` | 2.5 | Tolerância do filtro estatístico; maior preserva mais pontos. |
| `--oriented-box` | desligado | Usa OBB e estima o `yaw` do objeto. |
| `--config` | — | JSON com mapeamento de classes para objetos FlexSim. |

Esses valores dependem da densidade do scan real — calibre com uma cena conhecida.

## Limitações conhecidas (e caminhos de evolução)

- O backend heurístico continua disponível para cenas sintéticas e para
  máquinas sem PyTorch. Para o Warehouse Dataset, o backend recomendado é o
  PointNet++, que deve ser treinado com os scans locais antes da inferência;
  não há pesos pré-treinados distribuídos neste repositório.
- A segmentação PointNet++ é convertida em objetos por agrupamento espacial e
  caixa orientada. Em cenas muito densas ou com objetos encostados, calibre o
  `eps`, `min_points` e `score_threshold` usando IoU, F1 e erro dimensional.
- A correção do piso e a projeção BEV reduzem erros de pré-processamento, mas não transformam DBSCAN em um detector supervisionado. A calibração deve ser feita em uma sequência de validação, observando IoU, F1, erro de centro e número de clusters; um único scan não é suficiente para escolher `eps`.
- A bounding box alinhada aos eixos pode superestimar objetos em diagonal. A opção `--oriented-box` usa `get_oriented_bounding_box()` do Open3D e exporta o `yaw` estimado.
- Detecção é estática (um scan). Para operadores em movimento (gêmeo digital em tempo real), é preciso processar frames sequenciais e enviar posições via MQTT/OPC-UA ao FlexSim.

## Dados públicos para validação

- **Warehouse LiDAR Dataset** ([repositório oficial](https://github.com/anavsgmbh/lidar-warehouse-dataset), [Hugging Face](https://huggingface.co/datasets/Voxel51/lidar-warehouse-dataset)) — 3.287 scans de armazém com Velodyne VLP-16, 6.381 bounding boxes anotadas em 5 classes.
- **Semantic3D** — cenas externas grandes, usado pelo Open3D-PointNet++.

## Referências de projetos-base

- `AndresIslas99/pointcloud-object-detection` (GitHub) — detecção 3D em tempo real com DBSCAN para ambientes industriais (ROS2/Jetson).
- `isl-org/Open3D-PointNet2-Semantic3D` (GitHub) — segmentação semântica com deep learning.
- Prevu3D — referência comercial do fluxo reality capture → simulação (sem plugin FlexSim).
