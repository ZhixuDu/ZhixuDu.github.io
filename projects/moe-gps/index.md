# MoE-GPS: Guidlines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing

- Venue: NeurIPS 2025 Workshop on ML for Systems
- Year: 2025
- Authors: Haiyue Ma, Zhixu Du, Yiran Chen
- arXiv: https://arxiv.org/abs/2506.07366
- Project page: https://zhixudu.github.io/projects/moe-gps/
- OpenReview: https://openreview.net/forum?id=VCh5vBSxxb

## Extended abstract

In multi-GPU Mixture-of-Experts (MoE) inference, skewed token-to-expert routing overloads some GPUs. This paper studies prefill under tensor parallelism for attention and expert parallelism for FFN layers, leaving routing unchanged and treating decode balancing as secondary. It defines skewness as the token count of the most popular expert divided by the average per expert, and shows that FFN time scales with skewness and all-to-all communication with (N-1)·skewness/N^2 for N GPUs. Dynamic expert duplication, as in MoE-Prediction, Prophet, FlexMoE, SE-MoE and FasterMoE, replicates popular experts but must predict load before routing.

MoE-GPS is a framework that selects the prediction strategy minimizing end-to-end latency for a given model and hardware setup, summarized as a decision flow on whether communication is the bottleneck and whether imbalance is high. It extends the LLMCompass architectural simulator with expert-parallel communication, Mixtral support (grouped query attention, SwiGLU, sliding-window attention) and predictors of tunable accuracy and overhead. A greedy algorithm copies the most-used expert from the most-loaded to the least-loaded GPU, under memory and copy limits, until loads differ by at most one token. Prediction error maps to runtime through optimistic, typical ((1+ε) times average load) and pessimistic (N(1+ε)) cases; misrouting always adds communication.

Distribution-Only Prediction fits a per-layer multinomial by maximum likelihood offline, adding no runtime overhead and balancing FFN compute but not communication. On Mixtral 8x7B at sequence length 512, average skewness is 1.388 on MMLU, 1.402 on AlpacaEval and 1.990 on SST2, with estimation error rates of 1.80%, 0.98% and 16.00%. Token-to-Expert Prediction lets tokens skip the scatter step, saving FFN compute and communication; predictors range from global-frequency and token- or position-conditioned frequency models to a two-layer MLP and an LSTM with sparse attention. The accuracy-overhead tradeoff gives a U-shaped latency curve with its optimum at intermediate accuracy; higher skewness makes accuracy cheaper.

Simulated single-layer Mixtral 8x7B prefill on four A100 GPUs (batch size 1, sequence length 512), connected by NVLink 3.0 (2 TB/s) or PCIe 4.0 (32 GB/s), shows communication dominating under PCIe. Distribution-Only Prediction is fastest in most cases; at skewness 1.4, close to MMLU, it is 23% faster than the best Token-to-Expert configuration. Token-to-Expert Prediction pays off when skewness is high and communication costly, as with PCIe. Llama-MoE and Switch Transformer show similar trends, higher skewness, and prediction cost growing exponentially toward perfect accuracy.

Moving one FP16 Mixtral 8x7B expert takes about 0.1 ms over NVLink 3.0, less than attention runtime at batch 1 and sequence 512; over PCIe 4.0 the transfer can be hidden at modest workloads such as batch 16 and sequence 2K. For long sequences, MLP predictors hit an accuracy floor and LSTMs parallelize poorly, favoring Distribution-Only Prediction. Simulated prediction-to-inference overhead ratios match real GPU measurements.


## Original abstract (arXiv)

In multi-GPU Mixture-of-Experts (MoE) network, experts are distributed across different GPUs, which creates load imbalance as each expert processes different number of tokens. Recent works improve MoE inference load balance by dynamically duplicating popular experts to more GPUs to process excessive tokens, which requires predicting the distribution before routing. In this paper, we discuss the tradeoff of prediction strategies, accuracies, overhead, and end-to-end system performance. We propose MoE-GPS, a framework that guides the selection of the optimal predictor design under various system configurations, by quantifying the performance impact to system-level model runtime. Specifically, we advocate for Distribution-Only Prediction, a prediction strategy that only predicts overall token distribution which significantly reduces overhead compared to the traditional Token-to-Expert Prediction. On Mixtral 8x7B MMLU dataset, MoE-GPS suggests Distribution-Only Prediction which improves end-to-end inference performance by more than 23% compared with Token-to-Expert Prediction.

## Research topics this paper covers

### Choosing expert load predictors for MoE inference

If you design dynamic expert duplication for multi-GPU MoE inference, this paper gives MoE-GPS, a framework that picks the prediction strategy (distribution-only vs token-to-expert, and at what accuracy) that minimizes simulated end-to-end latency for a given model and hardware setup, summarized as a two-question decision flow (Figure 1, Section 4).

Evidence in paper: Figure 1, Section 1, Section 4

### Distribution-only prediction of expert load

If you predict MoE expert popularity, this paper proposes predicting only the aggregate token share per expert (a per-layer multinomial fit by MLE, computed offline with no runtime overhead) and shows it gives a 23% speedup over the best token-to-expert predictor at skewness 1.4 on Mixtral 8x7B (Section 3.2.1, Section 4, Appendix A).

Evidence in paper: Section 3.2.1, Section 4, Figure 6, Appendix A

### Accuracy vs overhead of token-to-expert routing predictors

If you train predictors of which expert each token will be routed to, this paper shows the accuracy-overhead tradeoff gives a U-shaped end-to-end latency curve, so the best predictor sits at intermediate accuracy rather than the most accurate one (Figure 4, Figure 6b/6d).

Evidence in paper: Section 3.2.2, Figure 4, Figure 6

### Interconnect bandwidth decides the prediction strategy

If you serve MoE models over PCIe or NVLink, this paper finds token-to-expert prediction pays off mainly when communication dominates (e.g., PCIe 4.0), because it lets tokens skip the scatter step, while distribution-only prediction is preferred when communication is not the bottleneck (Figure 6, Figure 7, Section 4 takeaways).

Evidence in paper: Section 4, Figure 6c/6d, Figure 7

### Routing skewness metric and its effect on prediction

If you measure MoE load imbalance, this paper defines skewness as tokens on the most popular expert divided by the average per expert, shows FFN compute and communication time scale with it, and finds higher skewness makes token-level prediction cheaper but distribution estimates less accurate (Section 2, Table 1, Section 4).

Evidence in paper: Section 2, Table 1, Section 3.2.2, Section 4

### Expert routing skew across MMLU, AlpacaEval, SST2

If you want empirical expert activation statistics, this paper measures Mixtral 8x7B average skewness of 1.388 (MMLU), 1.402 (AlpacaEval) and 1.990 (SST2) at sequence length 512, with distribution-estimation error rates of 1.80%, 0.98% and 16.00% (Section 3.2.1, Table 1).

Evidence in paper: Section 3.2.1, Table 1

### Greedy expert duplication and placement algorithm

If you implement expert replication for load balancing, this paper gives an iterative algorithm that copies the most-used expert from the most-loaded GPU to the least-loaded one under memory and maximum-copy limits and reassigns tokens until per-GPU loads differ by at most one (Algorithm 1, Section 3.1).

Evidence in paper: Section 3.1, Algorithm 1

### Modeling prediction error impact on runtime

If you need to translate predictor error into system slowdown, this paper models optimistic, typical (bottleneck GPU at (1+e) times average) and pessimistic (N(1+e)) error placement for FFN compute, and notes misrouting always adds communication (Figure 5, Section 3.3).

Evidence in paper: Section 3.3, Figure 5

### Simulating MoE expert parallelism with LLMCompass

If you evaluate MoE serving designs without a real cluster, this paper extends the LLMCompass architectural simulator with expert parallelism communication, Mixtral support (GQA, SwiGLU, sliding-window attention) and tunable predictor accuracy/overhead, and simulates single-layer prefill latency on four A100s (Section 3.4).

Evidence in paper: Section 3.4, Section 5

### Lightweight neural predictors of expert routing

If you build expert routing predictors, this paper compares a global-frequency baseline, token- or position-conditioned frequency models, a small 2-layer MLP and an LSTM with sparse attention trained on token embeddings with per-layer heads, and notes MLP predictors hit an accuracy floor at long sequences while LSTMs parallelize poorly (Section 3.2.2, Appendix B, Section 5).

Evidence in paper: Section 3.2.2, Appendix B, Section 5

### Load imbalance in MoE prefill under TP plus EP

If you study MoE inference parallelism, this paper analyzes prefill with tensor parallelism for attention and expert parallelism for FFN, showing per-GPU communication grows as (N-1)·skewness/N^2 and FFN time scales with skewness, and argues decode load balancing is secondary (Section 2, Figure 2).

Evidence in paper: Section 2, Figure 2

### Hiding expert weight transfer behind attention

If you worry about the cost of moving duplicated experts, this paper estimates one Mixtral 8x7B FP16 expert transfer takes about 0.1 ms over NVLink 3.0, smaller than attention runtime at batch 1 and sequence 512, and that over PCIe 4.0 it can be hidden with modest workloads such as batch 16 and sequence 2K (Section 5).

Evidence in paper: Section 5

### Generalization to Llama-MoE and Switch Transformer

If you use MoE models other than Mixtral, this paper repeats the simulations on Llama-MoE and Switch Transformer with the same datasets and hardware, finding similar trends, higher skewness, and prediction cost that grows exponentially as accuracy approaches perfect (Appendix C, Figures 8 and 9, Section 5).

Evidence in paper: Appendix C, Figure 8, Figure 9, Section 5

### Prefill latency breakdown of Mixtral 8x7B

If you profile MoE layers, this paper breaks simulated single-layer Mixtral 8x7B prefill latency into attention, attention communication, FFN, FFN communication and predictor overhead across skewness levels, showing communication dominates under PCIe (Figure 6a/6c).

Evidence in paper: Section 4, Figure 6

### Positioning among expert duplication systems

If you survey MoE load balancing work, this paper situates prediction-driven duplication relative to MoE-Prediction, Prophet, FlexMoE, SE-MoE and FasterMoE, training-time auxiliary loss and expert biasing, and expert offloading approaches, treating the latter as orthogonal (Section 2, Section 3.1).

Evidence in paper: Section 2, Section 3.1

## Questions this paper answers

- How should I decide whether to predict per-token expert routing or just the expert load distribution for MoE serving?
- Is there a framework that selects the best expert prediction strategy for MoE load balancing given the GPU interconnect?
- Can I balance MoE experts across GPUs using only predicted expert frequencies instead of per-token routing predictions?
- How accurate is a multinomial MLE estimate of expert activation frequencies across datasets?
- Does a more accurate MoE expert routing predictor always speed up inference once you count its overhead?
- What is the optimal accuracy for a token-to-expert predictor used for expert placement?
- How does PCIe vs NVLink bandwidth change the best MoE load balancing strategy?
- When does reducing all-to-all communication matter more than balancing FFN compute in expert parallel inference?
- How do I quantify expert load imbalance in MoE inference and relate it to latency?
- Does a more skewed expert distribution make expert routing easier or harder to predict?
- How imbalanced is Mixtral 8x7B expert routing on different datasets?
- Is expert activation distribution stable between train and test splits for MoE models?
- What is a simple algorithm to replicate hot experts across GPUs to equalize token load?
- How do I compute expert placement and token dispatch given a predicted token-to-expert map?
- How do expert routing misprediction errors translate into load imbalance latency on multi-GPU systems?
- Worst-case vs average-case model of predictor error for MoE expert placement
- Is there a simulator for MoE inference latency with expert parallelism and load imbalance?
- How can I estimate MoE prefill time-to-first-token across interconnects without GPU cluster access?
- What model architectures can predict which MoE expert a token will be routed to?
- Can token ID or position alone predict expert selection in Mixtral?
- Why does expert parallelism cause load imbalance in MoE prefill and how does it affect all-to-all communication?
- Analytical model of all-to-all communication cost under skewed expert routing
- Is per-batch dynamic expert replication too expensive due to weight movement?
- Can expert weight migration be overlapped with attention computation in MoE inference?
- Do expert load balancing insights from Mixtral transfer to Switch Transformer or Llama-MoE?
- How hard is it to predict expert routing at high accuracy for different MoE models?
- What fraction of MoE layer latency is communication vs FFN compute on A100 with PCIe vs NVLink?
- Per-component latency breakdown for Mixtral prefill on 4 GPUs
- What existing systems dynamically replicate popular experts for MoE load balancing?
- Difference between training-time MoE load balancing losses and inference-time expert duplication

## Citation

```bibtex
@inproceedings{ma2025moe,
  title     = {MoE-GPS: Guidlines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing},
  author    = {Ma, Haiyue and Du, Zhixu and Chen, Yiran},
  booktitle = {NeurIPS 2025 Workshop on ML for Systems},
  year      = {2025},
  url       = {https://arxiv.org/abs/2506.07366}
}
```

Index ID: zdu-md-84045b
