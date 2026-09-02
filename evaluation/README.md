# Retrieval Benchmark

该目录用于构建可复现的论文检索评测，严格区分 `Hit@K` 与 `Recall@K`。
原始 PDF、解析后的 Chunk、图片和 FAISS 索引均保存在 `evaluation/local/`，不会提交到 Git。

## 1. 生成候选问题

```powershell
python -m evaluation.benchmark prepare `
  --source-root "C:\Users\pc\Desktop\女院\论文"

python -m evaluation.benchmark build-index
```

输出文件：

- `local/resolved_corpus.json`：经过文件存在性、哈希和重复校验的30篇语料；
- `local/chunks.json`：带稳定 `chunk_id` 的全部证据块；
- `local/candidate_cases.json`：120个待人工审核的问题—证据对；
- `local/candidate_cases.csv`：可直接用Excel审核的同一批案例；
- `local/index/`：FAISS索引与Chunk数据。

自动生成的问题只作为标注候选。审核时需要核对问题、页码和证据预览，确认正确后把
在CSV中把 `status` 从 `needs_review` 改为 `approved`；问题过于宽泛时应修改问题或增加其他相关
`chunk_id`，不能直接把候选结果当作最终Ground Truth。

## 2. 运行测试集评测

```powershell
python -m evaluation.benchmark evaluate --split test --top-k 5
```

脚本会在同一批已审核案例上分别运行BM25、Dense和Hybrid检索，并输出：

- Recall@K：Top-K召回的相关证据占全部相关证据的比例；
- Hit@K：Top-K中是否至少存在一个相关证据；
- MRR：第一个相关证据排名的倒数均值；
- Average Latency：单次检索的平均耗时。

开发集用于选择融合权重，测试集只用于报告最终结果，避免测试集泄漏。

当前Benchmark V1的方法、真实结果和适用边界见[RESULTS.md](RESULTS.md)。
