# Swimba: Switch Mamba Model Scales State Space Models

- Venue: Advances in Neural Information Processing Systems 39 (NeurIPS 2026)
- Year: 2026
- Authors: Zhixu Du, Krishna Teja Chitty-Venkata, Murali Emani, Venkatram Vishwanath, Hai (Helen) Li, Yiran Chen
- arXiv: https://arxiv.org/abs/2603.06938
- Project page: https://zhixudu.github.io/projects/swimba/

## Extended abstract

Mixture-of-experts (MoE) is usually applied to feed-forward blocks, with a few methods such as SwitchHead and Mixture-of-Attention-Heads routing inside attention. This paper puts expert specialization inside the token mixer of selective state space models (SSMs) such as Mamba-2, keeping O(L) mixing cost without multiplying the dominant cost of the recurrent state update. It separates two designs that prior work conflates. MoE over separated SSMs keeps one state trajectory per expert, as in classical switching state-space models, so compute and state storage grow with the number of experts. MoE-parameterized SSM mixes experts in parameter space and evaluates a single recurrence. Hybrids such as BlackMamba, MoE-Mamba, Jamba, Zamba and Nemotron-H instead interleave dense SSM mixers with MoE MLPs, and Routing Mamba applies MoE to projections around the SSM core without this distinction.

Switch Mamba (Swimba) replaces the Mamba-2 in-projection with expert in-projections and a token-level softmax router. Experts produce the injection and readout streams (B, C, X), the transition A is shared across experts and time, and routing-weighted streams are summed into one input to the unchanged SSD kernel.

Five theorems support the design. The mixed layer is exactly one selective SSM with state size N, independent of the number of experts and of top-k sparsity. Its cost is O(T C_step + T C_mix) versus O(T E C_step) for separated experts. With a contractive transition (norm of A at most rho < 1) and bounded mixed streams, the layer is BIBO stable under token-wise expert switching. Mixed and separated models coincide when routing is time-invariant and readout shared; otherwise their output gap follows a stable recursion driven by each expert's deviation from the mixture injection. With linear expert projections and softmax routing, the layer strictly contains single-expert layers: two experts realize the logistic sigmoid, which a single-expert layer (a polynomial of degree at most 3 in this setting) cannot.

Swimba-14B replaces every Mamba-2 layer of the 52-layer Nemotron-H-8B hybrid with a 4-expert, top-1 Swimba layer, initializes from its checkpoint, freezes the backbone, and trains only experts and routers by supervised fine-tuning on the Llama-Nemotron Post-Training Dataset. Across ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, MMLU, OpenBookQA, PIQA, RTE and WinoGrande, average accuracy rises from 69.47 to 70.81 (normalized 70.98 to 72.34), with MMLU 71.7 to 75.0, while BoolQ and OpenBookQA dip slightly.

PyTorch profiling gives 1.51282e10 versus 1.51263e10 FLOPs per token, under 0.2% apart. In vLLM decoding on one 40GB A100 over batch sizes 1 to 64 and input-output lengths 256 to 2048, Swimba-14B is slightly slower, with as low as a 10% slowdown, a gap the paper attributes to routing overhead; adding experts at fixed active count should leave throughput and latency largely unchanged.


## Original abstract (arXiv)

Mixture-of-experts (MoE) is a common approach for increasing parameter capacity, but applying MoE to state space model (SSM) token mixers can multiply the cost of the recurrent state update. We study how to introduce expert specialization into selective SSMs while preserving computational efficiency. We show that MoE--SSM can refer to two designs: (1) MoE over separated SSMs, which maintains multiple state trajectories and thus scales compute with the number of experts; and (2) MoE-parameterized SSM, which mixes experts in parameter space, maintains a single state trajectory, and evaluates the recurrence once. Our method, Switch Mamba (Swimba), follows the second design by routing over expert-produced SSM streams. Theoretically, we establish well-definedness and stability for MoE-parameterized SSMs and characterize the relationship between the two designs. Empirically, we evaluate Swimba on standard benchmark tasks and measure real-time throughput and latency. Under matched FLOPs, Swimba achieves slightly better average performance than the baseline, with a small slowdown in real-time latency and throughput. Overall, these results suggest that parameter-space MoE can increase SSM capacity while keeping the dominant recurrence cost fixed.

## Research topics this paper covers

### Mixture-of-experts inside SSM token mixers

If you want to add MoE to the sequence-mixing layer of a state space model rather than to the FFN, this paper introduces Swimba, which replaces the Mamba-2 in-projection with a set of expert projections plus a token-level router (Section 4.1, Fig. 3).

Evidence in paper: Section 1, Section 4.1, Fig. 3

### Taxonomy of MoE-SSM designs

If you are comparing ways to combine MoE with state space models, this paper separates MoE over separated SSMs (one state trajectory per expert, compute grows with experts) from MoE-parameterized SSMs (experts mixed in parameter space, one trajectory) and argues prior work conflates them (Sections 1, 3.2-3.3, Fig. 2).

Evidence in paper: Section 3.2, Section 3.3, Fig. 2

### Single recurrence cost independent of expert count

If you worry that MoE in a recurrent layer replicates the scan, this paper proves the parameter-mixed layer costs O(T C_step + T C_mix) versus O(T E C_step) for separated experts, so the recurrence is evaluated once regardless of the number of experts (Theorem 2, Appendix A.2).

Evidence in paper: Section 3.4 Theorem 2, Appendix A.2

### Parameter-space mixing preserves SSD structure

If you need MoE routing to stay compatible with Mamba-2's SSD kernel, this paper shows that mixing expert injection and readout streams yields exactly one selective SSM with state size N, independent of the number of experts and of top-k sparsity, so the existing SSD implementation is reused unchanged (Theorem 1, Section 3.1).

Evidence in paper: Section 3.4 Theorem 1, Appendix A.1

### Stability of expert switching in recurrences

If you are concerned that token-wise expert switching destabilizes recurrent dynamics, this paper gives a BIBO stability bound: with a contractive shared transition (norm of A at most rho < 1) and bounded mixed injection and readout, state and output stay bounded (Theorem 3, Appendix A.3).

Evidence in paper: Section 3.4 Theorem 3, Appendix A.3

### When mixed and separated expert SSMs coincide

If you study whether averaging expert parameters approximates running separate expert recurrences, this paper proves exact equality when routing weights are time-invariant and readout is shared, and bounds the output gap otherwise by a stable recursion driven by each expert's deviation from the mixture injection (Theorem 4, Appendix A.4).

Evidence in paper: Section 3.4 Theorem 4, Appendix A.4

### Expressivity gain from routed linear projections

If you ask whether parameter-space MoE adds modeling power or is only an efficiency trick, this paper proves that with softmax routing over linear expert projections the layer strictly contains single-expert layers, via a two-expert construction that realizes the logistic sigmoid, which no single-expert layer (a degree-3 polynomial in this setting) can represent (Theorem 5, Appendix A.5).

Evidence in paper: Section 3.4 Theorem 5, Appendix A.5

### Shared transition, expert-specific B, C and X

If you are choosing which SSM parameters to make expert-specific, Swimba shares the transition A across experts and time and places expert diversity only in the injection and readout streams (B, C, X), which keeps one coherent state trajectory (Section 3.2, Eq. 7-9; Section 6).

Evidence in paper: Section 3.2 Eq. (7)-(9), Section 6

### Upcycling a pretrained hybrid Mamba-Transformer

If you want to add experts to an existing pretrained hybrid model cheaply, this paper initializes from the Nemotron-H-8B checkpoint, replaces every Mamba-2 layer with a Swimba layer in the 52-layer hybrid layout, freezes the copied backbone, and trains only new expert and router parameters with SFT on the Llama-Nemotron Post-Training Dataset (Sections 4.2-4.3).

Evidence in paper: Section 4.2, Section 4.3

### Matched-FLOPs accuracy on LM benchmarks

If you need evidence that SSM-layer MoE helps at constant compute, Swimba-14B (4 experts, top-1 per layer) raises the average accuracy from 69.47 to 70.81 (normalized 70.98 to 72.34) over Nemotron-H-8B across ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, MMLU, OpenBookQA, PIQA, RTE and WinoGrande, e.g. MMLU 71.7 to 75.0, while BoolQ and OpenBookQA dip slightly (Table 1).

Evidence in paper: Table 1, Section 5.1

### Per-token FLOPs of top-1 MoE SSM layers

If you measure the compute cost of sparse experts in SSM layers, this paper uses the PyTorch profiler and finds Swimba-14B at 1.51282e10 FLOPs per token versus 1.51263e10 for Nemotron-H-8B, a difference under 0.2% (Table 2, Section 5.2).

Evidence in paper: Table 2, Section 5.2

### vLLM serving cost of MoE-SSM models

If you deploy SSM or hybrid models in vLLM, this paper benchmarks decoding throughput and latency on a single 40GB A100 across batch sizes 1-64 and input-output lengths 256-2048, finding Swimba-14B slightly slower than Nemotron-H-8B, with as low as a 10% slowdown (Fig. 4, Fig. 5, Section 5.2).

Evidence in paper: Fig. 4, Fig. 5, Section 5.2

### FLOPs versus wall-clock gap from routing

If you study why MoE models run slower than their FLOPs suggest, this paper reports that despite near-identical FLOPs Swimba shows a larger real-time gap, which it attributes to routing overhead, and notes that adding experts with a fixed number of active experts should leave throughput and latency largely unchanged (Section 5.2).

Evidence in paper: Section 5.2

### Positioning against MoE-Mamba, BlackMamba, Jamba, Routing Mamba

If you survey MoE-SSM hybrids, this paper contrasts block-mixing approaches that interleave dense SSM mixers with MoE MLPs (BlackMamba, MoE-Mamba, Jamba, Zamba, Nemotron-H/Nemotron-3) and Routing Mamba's MoE projections, with its own parameter-space mixing that routes the B_t and C_t governing the state update (Section 6).

Evidence in paper: Section 6

### MoE for token mixing beyond attention

If you work on mixture-of-experts attention such as SwitchHead or Mixture-of-Attention-Heads, this paper adapts the idea of routing within token-mixing layers to linear-time SSMs, motivated by keeping O(L) mixing cost while growing parameters (Fig. 1, Section 6).

Evidence in paper: Fig. 1, Section 1, Section 6

### Link to switching state-space models

If you know classical switching state-space models (Ghahramani and Hinton), this paper relates them to MoE over separated SSMs, where each expert keeps its own memory, and contrasts this with compressing expert dynamics into one shared trajectory (Section 1, Section 3.4).

Evidence in paper: Section 1, Section 2.3, Section 3.4

## Questions this paper answers

- How can mixture-of-experts be applied to the Mamba token mixer instead of the MLP block?
- Is there an MoE version of Mamba-2 where the experts are inside the SSM layer?
- What is the difference between mixing expert outputs and mixing expert parameters in an MoE state space model?
- Does putting experts inside an SSM multiply the cost of the recurrent state update?
- Can you scale parameters of a Mamba layer with experts without running multiple scans?
- Computational complexity of mixture-of-experts recurrent state space layers
- Does routing between expert projections break the state space duality kernel in Mamba-2?
- How to keep a single hidden state trajectory when adding experts to a selective SSM
- Is a state space model with input-dependent expert routing BIBO stable?
- Stability guarantees for switching selective SSMs with time-varying input matrices
- Under what conditions is mixing parameters equivalent to mixing outputs of linear recurrences?
- Error bound between parameter averaging and output averaging in mixture of linear dynamical systems
- Does a mixture-of-experts projection strictly increase the expressivity of an SSM layer with one state update?
- Theoretical expressivity of soft routing over linear experts
- Which Mamba parameters should be routed by experts: the transition matrix or the input and output projections?
- Mixture of experts generating B and C matrices of a selective state space model
- Can you upcycle a pretrained Nemotron-H or Mamba-2 model into a mixture-of-experts SSM by training only experts and routers?
- Continued training of new MoE parameters on a frozen hybrid Mamba-Transformer backbone
- Does adding mixture-of-experts to Mamba layers improve MMLU and commonsense benchmarks at the same FLOPs?
- Benchmark results of a 14B MoE Mamba model versus Nemotron-H-8B
- How many FLOPs per token does a top-1 mixture-of-experts Mamba layer add?
- Measuring inference FLOPs of hybrid SSM language models with the PyTorch profiler
- What is the throughput and latency overhead of a mixture-of-experts Mamba model in vLLM?
- Decoding benchmark of hybrid Mamba-Transformer models on an A100 across batch sizes and sequence lengths
- Why does a mixture-of-experts model with equal FLOPs have higher latency than the dense baseline?
- Routing overhead in MoE inference for state space models
- How do MoE-Mamba, BlackMamba, Jamba and Routing Mamba differ in where they apply experts?
- Related work on mixture-of-experts for state space models and linear-time sequence models
- Is there an SSM analogue of mixture-of-attention-heads or SwitchHead?
- Increasing parameter count of linear-time sequence mixers with sparse experts
- How do switching state space models relate to mixture-of-experts in modern Mamba architectures?
- Trade-off between expert-specific memory and a shared hidden state in mixture of recurrent experts

## Citation

```bibtex
@inproceedings{du2026swimba,
  title     = {Swimba: Switch Mamba Model Scales State Space Models},
  author    = {Du, Zhixu and Chitty-Venkata, Krishna Teja and Emani, Murali and Vishwanath, Venkatram and Li, Hai and Chen, Yiran},
  booktitle = {Advances in Neural Information Processing Systems 39 (NeurIPS 2026)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2603.06938}
}
```

Index ID: zdu-md-84045b
