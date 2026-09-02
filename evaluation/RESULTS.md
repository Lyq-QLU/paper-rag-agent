# Retrieval Benchmark V1 Results

## Setup

- Corpus: 30 unique PDFs selected from multi-objective optimization, routing, home healthcare and scheduling literature.
- Index: 6,789 structure-aware chunks embedded with `paraphrase-multilingual-MiniLM-L12-v2` and indexed by FAISS `IndexFlatIP`.
- Development set: 30 evidence-grounded questions, used only to choose the Dense/BM25 fusion weight.
- Test set: 90 generated candidate questions. An independent evidence-sufficiency pass, which did not see retrieval rankings, accepted 84 and rejected 6 incomplete or contradictory figure/table cases.
- Query leakage control: questions do not contain the source PDF title, author, filename or page number.
- Selected fusion weight: Dense `0.60`, BM25 `0.40`, selected on development-set document Hit@5 and document MRR.

Questions are model-generated and independently evidence-verified, but have not yet received final human sign-off. Results should therefore be described as Benchmark V1 rather than a human-annotated gold benchmark.

## Locked Test Results

| Retrieval mode | Document Recall@5* | Document MRR | Evidence Chunk Hit@5 | Evidence Chunk MRR | Avg latency |
|---|---:|---:|---:|---:|---:|
| BM25 | 57.14% | 49.54% | 28.57% | 15.02% | 39.62 ms |
| Dense | 80.95% | 68.37% | 41.67% | 31.17% | 119.76 ms |
| Hybrid (0.60/0.40) | **85.71%** | 64.76% | 36.90% | 25.69% | 114.41 ms |

\* Each query has one relevant source paper, so document Recall@5 equals document Hit@5 under this benchmark definition.

## Interpretation

- Hybrid retrieval improves document Recall@5 by 4.76 percentage points over Dense retrieval and 28.57 points over BM25.
- Dense retrieval ranks the exact pre-labelled evidence chunk more effectively than the current Hybrid pipeline.
- The gap between document recall and exact-chunk hit shows that one relevant chunk per question is too narrow for evidence-level Recall@K. Equivalent answer-bearing chunks must be added before reporting strict evidence Recall@5.
- The previously drafted `Recall@5 89%` claim is not supported by this experiment and must not be used.

## Resume-safe wording

> 构建包含30篇论文、6,789个结构化Chunk和84个证据校验问题的检索评测集，对比BM25、Dense与Hybrid Retrieval；在无论文标题泄漏的测试集上，Hybrid文档级Recall@5达到85.7%，较Dense提升4.8个百分点。

Do not describe this dataset as manually annotated until a human reviewer has approved the questions and evidence labels.
