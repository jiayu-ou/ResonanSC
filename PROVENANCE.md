# 来源与整理范围

- `src/resonansc/core`、配置/数据适配：ResonanSC 本地 `sec3_all_donor_hvg/resonansc_integrated`，其中优化器包含该次修订的零 weight decay。
- `hierarchical.py`、`all_donor.py`、`rescue_alignment.py`：本地 `sec3_all_donor_hvg/run_hierarchical.py`、`all_donor_align.py`、`rescue_alignment.py`。路径改为显式参数；condition 层级原流程和阈值由示例配置记录。
- `post_alignment.py`：本地 `sec3_all_donor_hvg/compare_pairwise_union.py` 的 pairwise 模式，不含 GT 可视化或 union 模式；可复现实验中 166,845 细胞的输出标签。
- `pairwise_prepare.py`：本地 `sec3_pairwise_PWM_optimization/prepare.py`，改为显式路径、无数据集专属 cluster 数；`alignment_prepare.py` 是新输入桥接。
- `embedding/balanced_train.py`：本地 `sec3_WM20_umap/train_balanced_mmd.py`，保留 MMD200、16×256、无训练/测试集划分及平衡采样，移除数据集特定路径、写死细胞数和 LSI 维数；`embedding/lsi.py` 对应 `prepare_lsi.py`。
- 未知上游许可；见旧目录来源记录和原作者授权情况。此文件不是新的软件许可证。
