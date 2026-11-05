# ResonanSC Python API

Public source repository: https://github.com/jiayu-ou/ResonanSC

## Reviewer quick start (real PBMC data)

ResonanSC aligns single-cell RNA and ATAC batches, then supports pairwise label
resolution, P/W/M optimization, and a dual-modality embedding. The repository
contains the Python source, configuration templates, tests, and a small, real
paired subset of the 10x Multiome PBMC 10k data. The example below runs the
batch alignment on a CPU and checks that every input cell appears once in the
output. Python 3.10 or newer is required.

```bash
git clone https://github.com/jiayu-ou/ResonanSC.git
cd ResonanSC
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/run_pbmc_minimal.py
```

The result is `pbmc_minimal_run/alignment/cell_labels.csv` (800 cells in the
bundled subset). The script also writes its exact configuration to
`pbmc_minimal_run/config.yaml`. Use `--output PATH` for another new output
directory. The input files in `data/pbmc_minimal/` contain 400 paired barcodes
represented as 400 RNA and 400 ATAC rows, 500 shared expression genes, and
2,000 raw-count ATAC peaks. The subset was sampled with seed 2026 and the
feature selection used only the sampled matrices. Cell-type annotations were
excluded. To regenerate it from the prepared full PBMC matrices:

```bash
python data/pbmc_minimal/make_subset.py /path/to/resonansc_integrated
```

That directory must contain `expression.h5ad` and `peak_counts.h5ad` in the
format described below. The bundled inputs are sufficient for the quick-start
alignment; the full PBMC files and GTF are not needed for that command.
Optimization and embedding are separate, larger workflows described below.

Run the interface checks with
`python -m unittest discover -s tests -p 'test_*.py' -v`.
The real-data quick start verifies execution and cell accounting; it is not a
benchmark of biological accuracy. The source provenance is in
`PROVENANCE.md`. No software license is asserted for upstream code whose
redistribution terms have not been confirmed.

这是当前 ResonanSC RNA–ATAC 流程的独立 Python 包，公开分阶段 API；不依赖原实验目录；仅附可运行的小型 PBMC 示例数据，不打包完整实验数据或预训练模型。默认按 batch 初始化和合并、跨 batch 对齐，再跨模态对齐；也提供 sec3 使用的 condition 分层对齐作为可选模式，以及 pairwise DE 消歧义、P/W/M 优化和双模态 embedding。

## 安装

Python ≥3.10，建议使用独立环境：

```bash
python -m pip install -e .
# embedding 需要额外依赖与 CUDA
python -m pip install -e '.[embedding]'
```

训练 embedding 需要兼容 PyTorch 的 NVIDIA GPU 和 `scvi-tools==1.3.3`。P/W/M 优化在支持 PyTorch 的 CPU 或 GPU 上运行。**只加载可信的 PyTorch checkpoint**：其中的 pickle 可执行任意代码。

## 输入

- 对齐的 `expression_h5ad`：细胞 × 共同基因的 RNA 表达与 ATAC gene activity，细胞、基因名唯一；需要预先选好共同 HVG，列数与 `alignment.n_top_genes` 一致（当前 sec3 为 2,000）。`.obs['batch']` 区分以 `rna_` 和 `atac_` 开头的批次。默认模式不需要 condition 列。
- `peak_h5ad`：ATAC 细胞 × 共同 peak 的非负整数原始计数，barcode 唯一。准备 W 还需要同一基因组版本的 GTF。**不要把 gene activity 或 TF-IDF 当作原始 peak 计数。**
- RNA PCA 和 ATAC 原始 peak LSI 作为显式数组输入对齐 API，各需提供一份严格对应的 barcode 列表；这是对齐使用的特征，和后续 W-mapped gene activity 的 LSI 是不同步骤。当前实现使用特征矩阵的前 30 列。
- 选择 `mode="condition"` 时，condition 列每个批次必须唯一，且必须明确配置恰好两个 condition。`examples/latest_hierarchy.yaml` 是 sec3 使用的参数模板：RNA Leiden 0.9、ATAC 1.0；先 condition 内合并、全 donor 两两匈牙利匹配、condition 内再合并，再跨 condition 和模态对齐。条件组由 `alignment.condition_groups` 明确配置，不根据疾病名称或 donor 自动推断。请先核对 ATAC 与 RNA 每个批次的 condition，特别是实际数据中曾出现的标签反转。
- 同一批次内无法区分的类别是否合并取决于输入、相关性及阈值；对齐不使用外部 GT，也不宣称细胞亚型正确。全 donor 匈牙利步骤沿用本次 sec3 的实现：没有额外的相关性最低门槛，存在低相似度仍匹配的风险，需看输出的审计表。

## 1. 分层对齐

```python
import numpy as np
import resonansc as rs

# 修改 examples/latest_hierarchy.yaml 中的文件路径和基因组版本。
# 四个数组由你自己提供，行顺序必须与对应 barcode 完全一致。
rna_pca = np.load('rna_pca.npy')
rna_cells = np.load('rna_cell_names.npy', allow_pickle=False)
atac_lsi = np.load('atac_peak_lsi.npy')
atac_cells = np.load('atac_cell_names.npy', allow_pickle=False)
run_dir = rs.align('examples/latest_hierarchy.yaml',
                   rna_pca=rna_pca, rna_cell_names=rna_cells,
                   atac_lsi=atac_lsi, atac_cell_names=atac_cells)
```

默认先分别在每个 batch 中用 RNA PCA 或 ATAC peak LSI 初始化 Leiden 聚类，按各模态阈值合并相似簇；少于 50 个细胞的簇合并到最相似簇，50–99 个细胞的簇仅当相关性达到相应阈值时合并。随后同模态所有 batch 两两匈牙利对齐，最后跨模态对齐。

若需要重现 sec3 按 condition 初始化的流程，在上面的 `rs.align(...)` 中额外传入 `mode="condition"`；此模式沿用原先的 condition 内预合并、donor 对齐、condition 内再合并和跨 condition 对齐，**不执行默认模式的 batch 内小簇合并**。

输出目录必须是新目录。包含 `cell_labels.csv`、`hierarchical_alignment_checkpoint.pt`、各步骤和匈牙利匹配/救援审计表。对齐保持原细胞，不根据参考细胞类型强制修改标签。分辨率、阈值和输入空间的选择需结合数据复核，配置模板不是普适最优值。

## 2. 对齐后 pairwise DE 消歧义

输入为共享基因空间的 AnnData，RNA 行是原始计数，ATAC 行是 W 映射的非负 gene activity；`.obs` 含 `modality` 和 `final_alignment`。该过程只把 ATAC-only 原 cluster 整体映射到 RNA/ATAC 共享 cluster，RNA 标签不会更改。

```python
import anndata as ad
adata = ad.read_h5ad('rna_atac_common_gene_activity.h5ad')
labels, decisions, selected_genes = rs.resolve_labels(adata, margin=0.02, n_genes=100)
labels.index.name = 'cell_barcode'
labels['final_alignment'] = labels['pairwise_label']  # 仅修改副本供后续阶段使用
labels.to_csv('pairwise_labels.csv')
labels.reset_index().to_csv('pairwise_labels_for_embedding.csv', index=False)
decisions.to_csv('pairwise_decisions.csv', index=False)
```

各细胞先按 10,000 总量归一化并 log1p，再计算各 cluster 的平均特征和 Pearson 相关。前两名差距 ≤0.02 时，使用 RNA 两组之间 rank-sum 统计量绝对值最大的 100 个基因重新排序。默认强制映射最近候选、没有最低相关性保障；pairwise rank-sum 在这里是**选特征**，不是 donor 层面的差异表达推断。用于这一步的 W 应明示来源（已有可信优化 W 或明确的初始化），不要悄悄把它描述成新优化结果。

如果存在 RNA-only cluster，可显式传入 `allow_rna_only=True`；此时 ATAC-only cluster 以所有 RNA cluster 为候选，并保留原有 RNA 标签。仍无法匹配 ATAC 的 RNA-only cluster 不会获得逐类跨模态监督。

## 3. 构造 P/W/M 初值和优化

```python
# labels 是第 2 步返回的 DataFrame；省略 labels 可使用原始对齐标签。
init_checkpoint = rs.prepare_from_alignment(
    'examples/latest_hierarchy.yaml', alignment_dir=run_dir,
    labels=labels, label_key='pairwise_label')
```

此步骤需要在 YAML 中填写可用 GTF，以及相同的 `peak_genome_build`、`gtf_genome_build`。生成 `prepared/init_for_training.pt`、原始逐批训练数据、gene activity 和审计信息；P 使用对应标签 3/-5 logits，M 由 one-vs-rest marker 初始化，W 为 peak–gene 距离先验。若已有其他可信的初始 checkpoint 和训练数据，也可用 `rs.prepare_optimization(...)` 在保持原 W 初始化的同时换入 pairwise 标签并重建 M。

编辑 `examples/optimization_WM20.yaml` 的输入、输出和 CUDA 设备，然后：

```python
rs.optimize('examples/optimization_WM20.yaml')
```

模板只优化 W、M 各 20 轮，固定 P，和当前 WM20 运行一致；不自动声称比其他设置更优。输出包括每阶段 checkpoint、loss 和细胞标签。输出路径应设为新的独立目录，避免覆盖。

提取可信 checkpoint 中模型自己的加权 RNA pseudo-bulk：

```python
# annotation 按 cell barcode 建立唯一索引，至少含 batch 和 final_alignment
bulks, samples = rs.model_pseudobulk('output/optimization_WM20/train_M_1.pt', annotation)
```

该函数验证每个 batch 的细胞和 P 的 argmax 与标签一致，并直接返回保存的 `bulks_list`；不是原始 count 求和，不能直接作为 edgeR 的计数输入。矩阵特征集合就是模型输入基因，不自动覆盖全转录组。

## 4. 当前 W-LSI embedding

```python
import anndata as ad, numpy as np
rs.prepare_embedding('rna_counts.h5ad', 'pairwise_labels_for_embedding.csv',
                     'output/optimization_WM20/train_M_1.pt',
                     'output/hierarchy/prepared/training_multidata',
                     'output/embedding_input.h5ad')
a = ad.read_h5ad('output/embedding_input.h5ad')
lsi, atac_cells, lsi_model = rs.atac_w_lsi(a)  # pooled 31-SVD, 丢第1维，30维 sign-split ×10
np.save('output/atac_lsi.npy', lsi)
np.save('output/atac_cells.npy', atac_cells)
rs.train_embedding(input_path='output/embedding_input.h5ad',
    rna_map='rna_cluster_map.json', parents='atac_parent_sets.csv',
    atac_lsi_values='output/atac_lsi.npy', atac_lsi_names='output/atac_cells.npy',
    output='output/embedding', weight=200, latent=16, hidden=256)
```

`rna_cluster_map.json` 将 RNA cluster 整数 ID 映射为父类名；CSV 的列为 `atac_batch,atac_cluster,parent_candidates`，多个候选用 `|` 分隔。由使用者提供并审查父类，而非偷偷用 GT。默认 RNA/ATAC 各自预训练 100 轮，再共享分类头联合训练 50 轮；pretraining 使用所有细胞，不划测试集，联合训练按已知单标签分层平衡采样。只给定模糊父类集合或 Unknown 的细胞不会被该版本的平衡单标签采样器抽中；需根据实验需要检查这一限制。LSI 使用非负正负拆分输入 NB 解码器，是实验近似，不是严格的计数概率模型。若组间不存在共享标签、或存在空组，训练会报错，不能作为已成功对齐处理。

训练得到 `model_RNA.pt`、`model_ATAC.pt`、`X_latent.npy` 和 `embedding_results.h5ad`（`obsm['X_embedding']`）；不自动运行 UMAP。对同一组基因和已知批次的输入可用 `rs.infer_embedding(...)`，并需提供训练时相同 barcode 顺序的 ATAC LSI；不支持未知 batch 的直接投射。

## 验证与限制

`python -m unittest discover -s tests -p 'test_*.py' -v` 覆盖输入、细胞质量守恒、稀疏 rank-sum、标签不变、模型 pseudo-bulk、优化初始化和 LSI；`python tests/smoke_hierarchy.py` 用合成数据运行分层对齐→准备→W/M 各一轮。设置 `RESONANSC_TEST_GPU=1` 后，embedding 测试会运行 GPU 训练和推理各一次。这些是接口验证，不是新数据上的效果证明。真实 sec3 pairwise 解析在 166,845 个细胞上与已保存结果一致；模型 bulk 导出 2,000×288 与已保存矩阵最大误差约 2.4e-7；现有 MMD200 模型在 64 个细胞上重新推理平均绝对差约 1.6e-5。

核心模型与研究脚本来源记录在 `PROVENANCE.md`；未确认上游代码许可，不应据此推断可对外重新分发。
