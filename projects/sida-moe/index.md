# SiDA: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models

- Venue: Proceedings of the Seventh Conference on Machine Learning and Systems (MLSys 2024)
- Year: 2024
- Authors: Zhixu Du, Shiyu Li, Yuhao Wu, Xiangyu Jiang, Jingwei Sun, Qilin Zheng, Yongkai Wu, Ang Li, Hai (Helen) Li, Yiran Chen
- arXiv: https://arxiv.org/abs/2310.18859
- Project page: https://zhixudu.github.io/projects/sida-moe/
- OpenReview: https://openreview.net/forum?id=q26ydTFF5j
- Code: https://github.com/timlee0212/SiDA-MoE

## Extended abstract

Mixture-of-Experts (MoE) models add capacity without proportional compute, but serving them wastes GPU memory because most experts sit idle for any given input. On Switch Transformers, this paper finds that MoE layers hold 78% to 99% of parameters, that effective GPU memory utilization falls to about 5% on short SST2 sentences (roughly 24GB idle for Switch-base-128 and 50GB for Switch-base-256), and that despite load balancing a whole sentence activates fewer than 40% of experts in Switch-base-128 and fewer than 20% in Switch-base-256. At batch size 1, expert selection, invocation and communication take up to 72% of inference time on Switch-base-256, and the number of experts invoked dictates latency.

The paper presents SiDA-MoE (Sparsity-inspired Data-Aware), an inference system that keeps only needed experts on the GPU and the rest in host main memory. A hash-building thread runs an offline-trained predictor that maps a batch's embeddings to each token's expert and scaling factor at every MoE layer, while an inference thread processes the previous batch, loads predicted experts with pipelined transfers, and evicts others under a memory budget with a FIFO cache. Routers are removed from the forward pass; fixed routers such as Hash or BASE layers can serve as the predictor directly.

Corrupting or permuting other tokens on C4 with Switch-base-128 shows that a token's expert choice depends on only about 1 to 4 other tokens or positions. The predictor is therefore a 2-layer LSTM with SparseMax self-attention and a residual connection, trained with truncated knowledge distillation over the router's top-30 logits plus cross-entropy, since full distillation degraded the small predictor.

On Switch-base-8/64/128/256 with SST2, MRPC and MultiRC on an A100 80GB GPU, against Standard, DeepSpeed and Tutel, SiDA-MoE saves over 80% of GPU memory on SST2, falling to over 40% (Switch-base-128) and 20% (Switch-base-256) on 200 to 500 token MultiRC inputs. Throughput over the baseline average reaches 2.60x and 3.93x for Switch-base-128 and 256 on SST2 and 1.26x and 1.57x on MultiRC, but only 1.01x to 1.04x for Switch-base-8; inference latency falls by up to 72% (down to 28% of the baselines' on Switch-base-256), with larger gains on shorter inputs. Across budgets from about 10 to 80GB, expert offloading gives higher throughput than layer-wise model parallelism.

Top-3 expert prediction accuracy reaches 99.00% (Switch-base-8) and 98.78% (Switch-base-128) on SST2, about 90 to 92% on MultiRC. Downstream fidelity is 97.5 to 99% for Switch-base-8 and 92.6 to 93% for Switch-base-128; C4 perplexity rises from 6.68 to 18.49 for Switch-base-8 and 4.59 to 8.11 for Switch-base-256, so larger models tolerate misrouting better. The method is orthogonal to quantization, pruning and expert-caching schedulers, and the paper outlines GPU, main memory and SSD tiered offloading for models such as Switch-c-2048 (about 5TB) and a graph predictor using cross-layer activation dependencies.


## Original abstract (arXiv)

Mixture-of-Experts (MoE) has emerged as a favorable architecture in the era of large models due to its inherent advantage, i.e., enlarging model capacity without incurring notable computational overhead. Yet, the realization of such benefits often results in ineffective GPU memory utilization, as large portions of the model parameters remain dormant during inference. Moreover, the memory demands of large models consistently outpace the memory capacity of contemporary GPUs. Addressing this, we introduce SiDA-MoE ($\textbf{S}$parsity-$\textbf{i}$nspired $\textbf{D}$ata-$\textbf{A}$ware), an efficient inference approach tailored for large MoE models. SiDA-MoE judiciously exploits both the system's main memory, which is now abundant and readily scalable, and GPU memory by capitalizing on the inherent sparsity on expert activation in MoE models. By adopting a data-aware perspective, SiDA-MoE achieves enhanced model efficiency with a neglectable performance drop. Specifically, SiDA-MoE attains a remarkable speedup in MoE inference with up to $3.93\times$ throughput increasing, up to $72\%$ latency reduction, and up to $80\%$ GPU memory saving with down to $1\%$ performance drop. This work paves the way for scalable and efficient deployment of large MoE models, even with constrained resources. Code is available at: https://github.com/timlee0212/SiDA-MoE.

## Research topics this paper covers

### Serving large MoE models under limited GPU memory

If you work on deploying Mixture-of-Experts models whose parameters exceed GPU memory, SiDA-MoE keeps only predicted-active experts on the GPU and holds the rest in host main memory, reporting up to 80% GPU memory saving on Switch Transformers (Section 4.1, Figure 8).

Evidence in paper: Abstract; Section 1; Section 4.1; Figure 8

### Predicting expert activation before inference

If you work on expert prefetching for MoE, this paper trains an offline predictor (called a hash function) that maps the input embedding sequence to the experts each token will use at every MoE layer, so experts can be loaded ahead of time instead of after each router call (Sections 3.2-3.3).

Evidence in paper: Section 3.2 Challenge 1; Section 3.3; Algorithm 1

### Replacing MoE routers with a learned predictor

If you study MoE gating, SiDA-MoE removes the routers from the forward pass entirely (they are offloaded to main memory) and substitutes the predictor's expert choices and scaling factors, while noting fixed routers such as Hash layers or BASE layers can be plugged in directly (Sections 3.1, 3.3).

Evidence in paper: Section 3.1; Section 3.3

### Pipelining expert selection with inference

If you design MoE serving pipelines, the paper runs a hash-building thread that predicts expert tables for the next batch concurrently with the inference thread processing the current batch, with expert loading overlapped via pipeline parallelism and a FIFO expert cache under a memory budget (Section 3.1, Figure 5, Algorithm 1).

Evidence in paper: Section 3.1; Figure 5; Algorithm 1; Section 4 Implementation

### MoE inference overhead profiling

If you profile MoE inference cost, this paper measures that expert selection, expert invocation and communication take up to 72% of inference time on Switch-base-256 at batch size 1, growing with expert count, and notes that at small batch sizes the number of experts invoked dominates latency (Section 2.3, Figure 3, Remark 1).

Evidence in paper: Section 2.3; Figure 3; Remark 1

### Sentence-level expert activation sparsity

If you study expert usage patterns, the paper shows that even with load-balancing loss, a whole sentence activates fewer than 40% of experts in Switch-base-128 and fewer than 20% in Switch-base-256 on SST2, with idle ratios above 70% and 80% even for 80-token sentences (Section 2.4, Figure 4).

Evidence in paper: Section 2.4; Figure 4

### Effective GPU memory utilization of MoE

If you quantify memory waste in sparse models, the paper reports that MoE layers hold 78% to 99% of Switch Transformer parameters and that effective memory utilization can fall to about 5%, leaving roughly 24GB (Switch-base-128) and 50GB (Switch-base-256) idle on short SST2 inputs (Section 2.2, Table 2, Figure 2).

Evidence in paper: Section 2.2; Table 2; Figure 2

### Sparse cross-token dependency of routing decisions

If you analyze what determines MoE routing, the paper corrupts or permutes random fractions of other tokens on C4 with Switch-base-128 and uses a combinatorial estimate to infer that a token's expert choice depends on only about 1 to 4 other tokens or positions (Section 3.4.1, Eq. 2, Figures 6-7).

Evidence in paper: Section 3.4.1; Equation 2; Figure 6; Figure 7

### LSTM with sparsemax attention predictor

If you need a lightweight sequence model that focuses on few inputs, the expert predictor is a 2-layer LSTM with self-attention whose weights use SparseMax, plus a residual connection from the current token, motivated by the sparse dependency finding (Section 3.4.2).

Evidence in paper: Section 3.4.2

### Truncated knowledge distillation of router logits

If you distill a large teacher's softmax into a low-capacity student, the paper computes the KD loss only over the teacher's top-T logits (T=30) and adds a cross-entropy term, because full-distribution KD degraded the small predictor (Section 3.5, Section 4 Hyperparameters).

Evidence in paper: Section 3.2 Challenge 3; Section 3.5; Section 4

### Accuracy of expert activation prediction

If you need numbers on how predictable MoE routing is, top-3 hit rates of the predictor reach 99.00% (Switch-base-8, SST2) and 98.78% (Switch-base-128, SST2), dropping to about 90-92% on long MultiRC inputs (Section 4.6, Table 5).

Evidence in paper: Section 4.6; Table 5

### Accuracy cost of approximate routing

If you assess the quality cost of predicted routing, fidelity is 97.5-99% for Switch-base-8 but 92.6-93% for Switch-base-128 on SST2/MRPC/MultiRC, and C4 perplexity rises notably (e.g. 6.68 to 18.49 for Switch-base-8, 4.59 to 8.11 for Switch-base-256), with larger models more tolerant (Section 4.4-4.5, Tables 3-4).

Evidence in paper: Section 4.4; Section 4.5; Table 3; Table 4

### Throughput and latency versus DeepSpeed and Tutel

If you compare MoE inference systems, SiDA-MoE reports 2.60x and 3.93x throughput over the baseline average for Switch-base-128/256 on SST2, but only 1.01-1.04x for Switch-base-8, with gains shrinking on long MultiRC inputs (1.26x, 1.57x) and latency reduced to about 25% on SST2/MRPC and 60% on MultiRC (Section 4.2, Figures 9-10).

Evidence in paper: Section 4.2; Figure 9; Figure 10

### Expert offloading versus model parallelism under memory budgets

If you choose a strategy for a fixed GPU memory budget, the paper sweeps budgets from about 10 to 80GB on Switch-base-128/256 and finds data-aware expert offloading with a FIFO cache gives higher throughput than layer-wise model parallelism at every budget (Section 4.3, Figure 11).

Evidence in paper: Section 4.3; Figure 11

### Effect of input length on MoE memory savings

If you care how sequence length interacts with expert sparsity, memory reduction exceeds 80% on short SST2 sentences but falls to over 40% (Switch-base-128) and 20% (Switch-base-256) per the text for 200-500 token MultiRC inputs, since longer inputs touch more experts (Section 4.1, Figure 8).

Evidence in paper: Section 4.1; Figure 8; Figure 4

### Hierarchical GPU-CPU-SSD expert offloading

If you target trillion-scale MoE such as Switch-c-2048 (about 5TB), the discussion identifies host memory capacity as the limit and proposes GPU/main memory/SSD tiered expert offloading, plus a graph-structured predictor exploiting cross-layer activation dependencies (Section 6).

Evidence in paper: Section 6

### Positioning among MoE serving systems

If you survey efficient MoE inference, the related work contrasts SiDA-MoE's data-aware look-ahead prediction with communication-focused systems (DeepSpeed-MoE, Tutel, SE-MoE), task-enforced sparsity (M3ViT, Edge-MoE) and expert-caching schedulers (EdgeMoE, dynamic expert swapping), and states it is orthogonal to quantization and pruning (Section 5).

Evidence in paper: Section 5; Table 1

## Questions this paper answers

- How can I run a Mixture-of-Experts model that does not fit in a single GPU's memory?
- Methods for reducing GPU memory footprint of MoE inference by keeping experts in CPU RAM
- Can you predict which MoE experts a token will activate before running the model?
- Look-ahead expert prefetching for mixture-of-experts inference
- Can a small network approximate the router of a trained Switch Transformer?
- What happens to MoE accuracy if the gating function is replaced by an external predictor?
- How to overlap MoE expert routing and CPU-GPU expert transfer with computation?
- Two-thread producer-consumer design for mixture-of-experts serving
- What fraction of MoE inference latency is spent on routing and expert dispatch?
- Why is mixture-of-experts inference slow at batch size 1?
- How many experts does a single input sequence actually use in a Switch Transformer?
- Is expert activation sparse at the sequence level despite load balancing?
- How much GPU memory do inactive experts waste in MoE inference?
- Memory breakdown of Switch Transformer parameters between expert and dense layers
- Does a token's expert routing depend on the rest of the sequence after self-attention?
- Which context tokens influence MoE router decisions?
- Lightweight architectures for predicting MoE routing from token embeddings
- Using sparsemax attention in an LSTM to focus on a few critical tokens
- Top-k truncated knowledge distillation for a small student that cannot match the full teacher distribution
- How to train a predictor to approximate MoE gating scaling factors?
- How accurately can expert routing be predicted in Switch Transformers?
- Does expert prediction accuracy drop for longer input sequences?
- How much does perplexity degrade when MoE experts are misrouted?
- Are larger MoE models more robust to wrong expert selection?
- Benchmarks of MoE inference frameworks such as DeepSpeed-MoE and Tutel on Switch Transformers
- Does MoE inference speedup from expert offloading depend on sequence length and number of experts?
- Throughput of MoE inference as a function of GPU memory budget
- Expert caching versus layer offloading for memory-constrained MoE serving
- Do longer prompts reduce the benefit of expert offloading in MoE models?
- Relationship between sequence length and number of activated experts
- Offloading MoE experts to SSD for models larger than CPU memory
- Do expert activations in one MoE layer predict those in the next layer?
- Survey of systems for efficient mixture-of-experts inference and expert offloading
- Can expert prediction be combined with quantization or expert caching policies for MoE?

## Citation

```bibtex
@inproceedings{du2024sida,
  title     = {SiDA: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models},
  author    = {Du, Zhixu and Li, Shiyu and Wu, Yuhao and Jiang, Xiangyu and Sun, Jingwei and Zheng, Qilin and Wu, Yongkai and Li, Ang and Li, Hai and Chen, Yiran},
  booktitle = {Proceedings of the Seventh Conference on Machine Learning and Systems (MLSys 2024)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2310.18859}
}
```

Index ID: zdu-md-84045b
