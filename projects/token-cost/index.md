# What Does a Token Cost? A Mixture-of-Agents Measurement of Sufficient Per-Token Compute

- Venue: Under review
- Year: 2026
- Authors: Zhixu Du, Weijia Han, Hai (Helen) Li, Yiran Chen
- arXiv: https://arxiv.org/abs/2610.02491
- Project page: https://zhixudu.github.io/projects/token-cost/

## Extended abstract

This paper asks how much computation each generated token requires. Large language models spend the same computation on every token, and speculative decoding, routing, and adaptive-depth methods assume much of it is unnecessary, yet per-token requirements have not been measured. The paper measures them through a token-level Mixture-of-Agents (MoA) lens: fifteen models from Qwen2.5, OLMo-2, and DeepSeek-R1-Distill-Qwen each try to reproduce a checker-verified reference token by token, conditioned on the correct preceding tokens, and the architecture-derived FLOPs of the cheapest agent that succeeds define the token's sufficient compute, which upper-bounds what the token requires. Casting token-level MoA as a Markov decision process, the paper shows that the minimum-cost selector for a fixed reference picks the cheapest matching agent at each position, so the summed map is the minimum reproduction cost and bounds the cost of any correct response.

On GSM8K, MATH-500, and HumanEval, a 0.5B agent reproduces 92-95% of reference tokens, and across the Qwen, OLMo, and R1-distilled panels the most expensive 10% of tokens account for 64-80% of estimated FLOPs. Allocation remains uneven when unresolved tokens are excluded rather than priced at twice the largest model's cost, with resolved-only Gini coefficients from 0.218 to 0.547. Families also share an ordering of expensive positions: after controlling for surprisal and position, partial Spearman correlations among Qwen2.5, OLMo-2, and Falcon3 range from 0.322 to 0.398 on Qwen-written and 0.417 to 0.475 on Llama-written references. The association is positive on all twelve benchmarks (50,019 prompts across math, code, question answering, knowledge, and commonsense), moves by less than 0.05 under alternative success criteria, a second verified response, memorization screening, and GSM-Symbolic, reaches 0.520 on text from a Qwen3-30B-A3B mixture-of-experts generator, and exceeds greedy or soft speculative-rejection signals (0.15-0.38). Digits, operators, word length, frequency, and position explain only 2.5-3.2% of ranked variance, and reproduction indices correlate 0.209 to 0.213 with FlexiDepth's learned layer allocations.

Cheap reproduction does not ensure reliable continuation. When a 0.5B model generates from a verified midpoint before Qwen-72B resumes, the accuracy gap grows from -0.09 points at one token to -9.44 points at 64 tokens, against -1.50 points for a 7B model. Completing the whole suffix costs the 0.5B model 54.08 points from the prompt and 37.64 from the midpoint, and reference cost weakly predicts handover loss (Spearman 0.195).

On all 500 MATH-500 problems, the oracle map of 0.5B agreement with a 72B reference helps routing reach 69.6% accuracy at a projected 5.12 seconds, compared with 68.4% and 7.59 seconds for the best confidence-routing baseline. Setting draft lengths from runs of matching positions uses 32.6% fewer draft tokens than a fixed 16-token window with similar accepted tokens, and about 20% lower projected latency (3.99 versus 4.98 seconds) at similar accuracy; the reduction ranges from 8.3% to 24.7% across 36 timing scenarios. Latencies are projected from published Qwen2.5 vLLM benchmarks, and a logical FLOP ledger shows drafting spending more FLOPs than the 72B model alone. These comparisons reveal remaining allocation headroom, motivating controllers that exploit sufficient-compute structure without advance reference access.


## Original abstract (arXiv)

Large language models spend the same amount of computation on every token they generate, regardless of how difficult each token is to produce. Methods such as speculative decoding and model routing are built on the premise that much of this computation is unnecessary, yet the computation an individual token actually requires has not been measured. We measure it through a Mixture-of-Agents (MoA) lens: a panel of fifteen language models of increasing capacity, drawn from three families, in which every agent attempts to reproduce a reference sequence token by token, conditioned on the correct preceding tokens. We define the inference cost of the smallest agent that succeeds as the token's sufficient compute, which upper-bounds what the token requires. On three core benchmarks, a 0.5B agent reproduces 92--95\% of reference tokens. Across Qwen, OLMo, and R1-distilled panels, the most expensive 10\% account for 64--80\% of estimated FLOPs. On all 500 MATH-500 problems, the MoA-derived map helps model routing reduce projected latency from 7.59 to 5.12 seconds while slightly improving accuracy, relative to the best confidence-routing baseline. The MoA-map helps drafting use 32.6\% fewer draft tokens and approximately 20\% lower projected latency than fixed-window drafting at similar accuracy. These comparisons reveal remaining allocation headroom, motivating controllers that exploit sufficient-compute structure.

## Questions this paper answers

### How much computation does each individual token in an LLM response actually need?

If you work on adaptive or efficient inference, this paper defines a token's sufficient compute as the FLOPs of the cheapest panel model whose prediction reproduces that token given the correct prefix, giving an observed upper bound on what the token requires (Section 2, Definition 1).

Topic: Measuring per-token compute requirements of LLMs. Evidence in paper: Section 2, Definition 1.
Also asked as: Is there a way to measure which generated tokens could have been produced by a much smaller model?

### How can choosing among several LLMs at each decoding step be formulated as an MDP?

If you study model selection during decoding, this paper casts token-level mixture-of-agents as a Markov decision process under a compute budget and shows via a Bellman recursion that, along a fixed verified reference, the minimum-cost controller simply picks the cheapest matching agent at each position (Section 3, Eqs. 3-9, Algorithm 1).

Topic: Token-level mixture-of-agents as decision process. Evidence in paper: Section 3, Eqs. 3-9, Algorithm 1, Figure 2.
Also asked as: What is the optimal per-token model selector when the target output is known in advance?

### Does the cost of reproducing one correct answer bound the cost of producing any correct answer?

If you need formal guarantees for oracle compute measurements, this paper proves that summed per-token sufficient compute equals the minimum cost over selectors that reproduce a reference exactly and upper-bounds the cost of producing any correct response, and gives a local witness bound for larger model classes (Appendix B.2, Eq. 15).

Topic: Upper-bound semantics of reference reproduction cost. Evidence in paper: Appendix B.2.
Also asked as: What assumptions make greedy per-token model selection cost-optimal?

### What fraction of a large model's output tokens can a 0.5B model predict correctly under teacher forcing?

If you work on small/large model collaboration, this paper finds that Qwen2.5-0.5B reproduces 91.9-95.2% of Qwen-72B reference tokens on GSM8K, MATH-500, and HumanEval, with OLMo-2 1B and R1-Distill-Qwen 1.5B at 89.34-91.74% and 89.80-92.27% (Section 4.1, Figures 1 and 3).

Topic: Small models reproduce most reference tokens. Evidence in paper: Section 4.1, Figures 1 and 3.
Also asked as: How often do small and large LLMs agree on the next token in math and code responses?

### Is the compute needed for LLM generation concentrated in a few hard tokens?

If you study where inference compute is spent, this paper shows the most expensive 10% of tokens account for 63.66-79.54% of estimated FLOPs in the Qwen panel, 63.75-69.37% in OLMo, and 67.19-74.76% in R1-Distill-Qwen, with the top 1% reaching 52.94-60.14% for Qwen (Section 4.1, Figure 3).

Topic: Heavy-tailed concentration of per-token compute. Evidence in paper: Section 4.1, Figure 3, Table A6.
Also asked as: What share of FLOPs do the hardest 10% of tokens require?

### How sensitive are token compute concentration measurements to tokens no model in the panel can reproduce?

If you worry that concentration results depend on how unmatched tokens are priced, this paper reports resolved-only top-10% shares of 51-60% (Qwen2.5), 40-44% (OLMo-2), and 26-32% (R1-Distill-Qwen) and resolved-only Gini coefficients from 0.218 to 0.547, with unresolved rates of 0.14-8.72% (Appendix C.1, Tables A5 and A6).

Topic: Cost conventions for unresolved tokens and Gini. Evidence in paper: Appendix C.1, Tables A5, A6.
Also asked as: What Gini coefficient describes the inequality of per-token compute in LLM outputs?

### Do different LLM families find the same tokens hard to predict?

If you study whether token difficulty is model-specific or shared, this paper finds that Qwen2.5, OLMo-2, and Falcon3 agree on which word positions need larger models, with partial Spearman correlations of 0.322-0.398 on Qwen-written and 0.417-0.475 on Llama-written references after controlling for surprisal and position (Section 4.1, Table 1).

Topic: Shared token difficulty ordering across model families. Evidence in paper: Section 4.1, Table 1, Figure 6.
Also asked as: Is per-token difficulty in generated text consistent across model families beyond surprisal?

### Is cross-model agreement on hard tokens explained by memorization or benchmark contamination?

If you want to know whether shared difficulty survives confounds, this paper shows the association is positive on all twelve benchmarks and moves by less than 0.05 under extra controls, top-five and p>=0.5 criteria, a second verified response, memorization screening (33 of 1,758 items flagged), and GSM-Symbolic, and reaches 0.520 on Qwen3-30B-A3B MoE-generated text (Section 4.1, Appendix C.2, Tables A3 and A4).

Topic: Robustness checks on cross-family agreement. Evidence in paper: Appendix C.2, Tables A3, A4.
Also asked as: Does shared token difficulty hold across math, code, QA, and commonsense benchmarks?

### Is small-versus-large model disagreement a good proxy for token difficulty across families?

If you work on speculative decoding acceptance, this paper compares its reproduction indices with greedy and soft rejection proxies between each family's smallest and largest models and finds the reproduction ordering more shared across families (0.322/0.417 versus 0.150/0.199 greedy and 0.179/0.384 soft) (Section 4.1, Appendix C.2).

Topic: Reproduction difficulty versus speculative acceptance signals. Evidence in paper: Section 4.1, Appendix C.2.
Also asked as: How consistent are speculative decoding rejection positions across different model families?

### Can token type, such as numbers or punctuation, predict which tokens need a larger model?

If you hope simple lexical cues identify hard tokens, this paper finds longer words weakly associated with higher reproduction demand and digits and operators weakly negative, with regressions on digit/operator indicators, word length, frequency, and position explaining only 2.5-3.2% of ranked variance (Section 6, Appendix C.2).

Topic: Surface features poorly predict expensive tokens. Evidence in paper: Section 6, Appendix C.2.
Also asked as: Which lexical features explain per-token compute demand in LLM reasoning text?

### If a small model agrees with a large model token by token, can it safely generate several tokens on its own?

If you study exposure bias or small-model handoffs, this paper shows that cheap reproduction under the correct prefix does not ensure reliable continuation: letting 0.5B generate before Qwen-72B resumes moves accuracy from -0.09 pp at one token to -9.44 pp at 64 tokens, and reference cost predicts handover loss only weakly (Spearman 0.195) (Section 4.2, Figure 4a, Table A8, Appendix D).

Topic: Teacher-forced agreement versus free-running continuation. Evidence in paper: Section 4.2, Figure 4a, Table A8, Appendix D.
Also asked as: How quickly does accuracy degrade when a small LLM takes over generation for a few tokens?

### How much accuracy does a small model lose when it finishes a reasoning chain started by a large model?

If you work on model relay or partial-solution handoff, this paper measures accuracy when 0.5B to 32B models complete a verified prefix from character fractions 0 to 0.9: 0.5B loses 54.08 points from the prompt and 37.64 from the midpoint, while 7B loses 1.50 points after 64 delegated tokens and 32B stays near zero (Section 4.2, Figure 4b, Table A9).

Topic: Mid-solution handover by capacity and start point. Evidence in paper: Section 4.2, Figure 4, Tables A8, A9.
Also asked as: Does handing over later in a solution reduce the accuracy cost of switching to a smaller model?

### How much headroom is left in confidence-based small/large token routing?

If you build token-level routers or cascades, this paper gives a routing controller oracle access to positions where 0.5B matches a 72B reference on all 500 MATH-500 problems, reaching 69.6% accuracy at 5.12 projected seconds versus 68.4% and 7.59 seconds for the best confidence-threshold cascade (theta = 0.99) (Section 4.3, Figure 5a, Tables A10 and A11).

Topic: Oracle token routing versus confidence cascades. Evidence in paper: Section 4.3, Figure 5a, Tables A10, A11.
Also asked as: What would an oracle router that knows token agreement achieve on MATH-500?

### How much speedup is left if speculative decoding knew the ideal draft length per block?

If you tune draft length in speculative decoding, this paper sets each draft block to the run of positions where 0.5B matches the 72B reference, proposing 301,691 versus 447,731 draft tokens (32.6% fewer) with similar accepted tokens and lowering projected latency from 4.98 to 3.99 seconds at 69.4% versus 69.0% accuracy (Section 4.3, Figure 5b, Appendix E.1).

Topic: Choosing speculative draft length from agreement maps. Evidence in paper: Section 4.3, Figure 5b, Appendix E.1, Table A10.
Also asked as: Do fixed-window drafts waste many rejected tokens compared with an oracle draft length?

### How do you count per-token inference FLOPs for a transformer with a KV cache?

If you need a per-token FLOP model for dense and MoE transformers, this paper counts attention, projection, gated feed-forward, router, and vocabulary matrix multiplications with prefix-dependent attention cost, and its ledger shows drafting policies using more logical FLOPs (112.07 TFLOPs for map drafting, 154.30 for fixed) than 72B alone (101.92) despite lower projected latency (Appendix B.1, Table A10).

Topic: Architecture-derived FLOP accounting for decoding. Evidence in paper: Appendix B.1, Eqs. 12-14, Table A10.
Also asked as: Why can speculative decoding lower latency while increasing total FLOPs?

### How can one project LLM decoding latency from published tokens-per-second benchmarks?

If you need latency estimates without a production deployment, this paper fits per-call and per-position costs to official Qwen2.5 vLLM A100 benchmarks (maximum relative error 2.13%) and tests 36 timing scenarios, under which map drafting stays faster than fixed-window and adaptive drafting by 8.3-24.7% and 7.3-18.4% (Appendix E.2).

Topic: Latency projection from published throughput benchmarks. Evidence in paper: Appendix E.2, Eqs. 18-19.
Also asked as: How sensitive are speculative decoding speedup estimates to timing assumptions?

### Do layer-skipping models spend more depth on the tokens that need larger models?

If you study adaptive-depth methods, this paper compares its reproduction indices with FlexiDepth's learned layer allocations on 600 generations, finding pooled partial correlations of 0.209 (Qwen panel) and 0.213 (OLMo panel), ranging from 0.061 on copy to 0.405 on product tasks (Appendix C.3, Table A7).

Topic: Agreement with learned layer-skipping allocations. Evidence in paper: Appendix C.3, Table A7.
Also asked as: How does learned adaptive depth relate to cross-model token difficulty?

### Is there a study scoring many model sizes against verified responses on math, code, and QA benchmarks?

If you need a checker-verified token-level dataset across domains, this paper builds references from Qwen2.5-72B and Llama-3.3-70B on twelve benchmarks with 50,019 prompts spanning GSM8K, MATH-500, HumanEval, LiveCodeBench, MBPP+, AIME 2024, ARC-Challenge, BBH, MMLU, MMLU-Pro, HellaSwag, and NQ-Open, scored against Qwen2.5, OLMo-2, R1-Distill-Qwen, and Falcon3 panels (Section 4, Appendix A, Tables A1 and A2).

Topic: Verified-response corpus across twelve benchmarks. Evidence in paper: Section 4, Appendix A, Tables A1, A2.
Also asked as: How do reasoning-distilled models compare with their base family on token-level reproduction?

## Citation

```bibtex
@misc{du2026token,
  title         = {What Does a Token Cost? A Mixture-of-Agents Measurement of Sufficient Per-Token Compute},
  author        = {Du, Zhixu and Han, Weijia and Li, Hai and Chen, Yiran},
  year          = {2026},
  eprint        = {2610.02491},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.02491}
}
```

Index ID: zdu-md-84045b
