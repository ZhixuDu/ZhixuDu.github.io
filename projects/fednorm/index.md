# Rethinking Normalization Methods in Federated Learning

- Venue: Proceedings of the 3rd International Workshop on Distributed Machine Learning (DistributedML '22)
- Year: 2022
- Authors: Zhixu Du, Jingwei Sun, Ang Li, Pin-Yu Chen, Jianyi Zhang, Hai (Helen) Li, Yiran Chen
- arXiv: https://arxiv.org/abs/2210.03277
- Project page: https://zhixudu.github.io/projects/fednorm/
- DOI: https://doi.org/10.1145/3565010.3569062
- Code: https://github.com/ZhixuDu/FedNorm

## Extended abstract

This paper explains why normalization layers matter in federated learning (FL) and why batch normalization (BatchNorm) degrades the FedAvg global model. It identifies external covariate shift: because devices run independent multi-step local training, per-channel feature statistics diverge across devices and neurons must re-adapt to a new feature distribution after every aggregation, which slows global convergence.

A fully connected layer analysis with whitened inputs shows that external covariate shift amounts to a shift in weight norms: when one client's weights have a much larger norm, averaging lets it dominate and obliterates the contribution of clients with small-norm weights. Normalization layers counter this through their scale-invariance: the gradient of a scale-invariant BatchNorm or LayerNorm layer is orthogonal to the preceding weights, so the squared weight norm grows only as the initial norm plus accumulated squared gradients, keeping the norms of clients that share an initialization balanced.

The paper argues that BatchNorm fails in FL not only because of non-IID data but because the same channel is trained to have different statistics on different devices, so averaging local running statistics on the server yields statistics that mismatch the aggregated model's features. A toy MNIST experiment supports this: two 3-layer, single-channel convolutional models with BatchNorm, trained from the same initialization on data that differ only in mean, end up with clearly different output statistics in the same channel. LayerNorm normalizes each sample across channels, keeps no cross-device running statistics, and is proposed as the replacement. The paper contrasts this explanation with FedBN (personalized FL only), GroupNorm, and HeteroFL's static BatchNorm.

Experiments run FedAvg on CIFAR-10 split pathologically across 100 devices (2 classes, 100 samples per class each), 10 devices per round, 10 local epochs, batch size 64, SGD with learning rate 0.01, and 5000 rounds over three seeds on TITAN RTX GPUs, with VGG-11, ResNet-18 and a simple CNN. It compares group, layer, instance, batch, fixed (non-aggregated) batch, and no normalization. LayerNorm converges fastest and reaches the best or near-best accuracy: 63.52% on VGG-11 (11.96 and 27.01 points above BatchNorm and no normalization), 59.70% on ResNet-18 (BatchNorm 34.38%), and 52.02% on the CNN (BatchNorm 33.29%). GroupNorm tracks LayerNorm closely (63.10% on VGG-11), InstanceNorm trails it on all three models, and fixed BatchNorm does not help (33.85% on ResNet-18). ResNet-18 without normalization reaches 60.73%, attributed to residual connections keeping weights near initialization, but needs careful hyperparameter tuning for stable training.


## Original abstract (arXiv)

Federated learning (FL) is a popular distributed learning framework that can reduce privacy risks by not explicitly sharing private data. In this work, we explicitly uncover external covariate shift problem in FL, which is caused by the independent local training processes on different devices. We demonstrate that external covariate shifts will lead to the obliteration of some devices' contributions to the global model. Further, we show that normalization layers are indispensable in FL since their inherited properties can alleviate the problem of obliterating some devices' contributions. However, recent works have shown that batch normalization, which is one of the standard components in many deep neural networks, will incur accuracy drop of the global model in FL. The essential reason for the failure of batch normalization in FL is poorly studied. We unveil that external covariate shift is the key reason why batch normalization is ineffective in FL. We also show that layer normalization is a better choice in FL which can mitigate the external covariate shift and improve the performance of the global model. We conduct experiments on CIFAR10 under non-IID settings. The results demonstrate that models with layer normalization converge fastest and achieve the best or comparable accuracy for three different model architectures.

## Research topics this paper covers

### External covariate shift in federated learning

If you study why FedAvg global models converge slowly, this paper names and defines external covariate shift: per-channel feature statistics diverge across clients because each runs independent local training, so neurons must re-adapt after every aggregation (Section 1, Section 3.1, Figure 1).

Evidence in paper: Section 1; Section 3.1; Figure 1

### Why batch normalization fails in federated learning

If you work on BatchNorm in FL, this paper argues the accuracy drop is not only due to non-IID data: clients' channel statistics drift apart during local training, so averaged running statistics on the server mismatch the aggregated model's real feature statistics (Section 3.3).

Evidence in paper: Section 3.3

### Layer normalization as BatchNorm replacement in FL

If you are choosing a normalization layer for a federated CNN, this paper recommends LayerNorm because it normalizes per sample and has no cross-client running statistics; it gives the fastest convergence and best or near-best CIFAR-10 accuracy on CNN, VGG-11 and ResNet-18 (Section 3.4, Table 2, Figure 4).

Evidence in paper: Section 3.4; Table 2; Figure 4

### Weight-norm imbalance obliterating client contributions

If you study aggregation fairness or client contribution in FedAvg, this paper shows with an FC-layer analysis that clients whose weights have much larger norms dominate the averaged model, erasing the contribution of small-norm clients (Section 3.1, Figure 2).

Evidence in paper: Section 3.1; Figure 2

### Scale invariance keeps client weight norms balanced

If you work on theory of normalization and auto learning-rate tuning, this paper uses the scale-invariance of BN/LN (gradient orthogonal to weights, Lemma 1) to show weight norms grow as initial norm plus accumulated squared gradients, so clients starting from the same initialization keep comparable norms (Section 2.2, Section 3.2, Eq. 5 and 7).

Evidence in paper: Section 2.2; Section 3.2; Lemma 1; Eq. 5, 7

### Two-device toy experiment on channel statistics

If you want direct evidence of client feature-statistics divergence, this paper trains two identical 3-layer single-channel conv models with BatchNorm on MNIST from the same initialization, with data differing only in mean, and shows the same channel ends up with clearly different output statistics on each device (Section 3.3, Figure 3).

Evidence in paper: Section 3.3; Figure 3

### Comparing BN, GN, LN, IN and no normalization in FedAvg

If you need a benchmark of normalization layers under FedAvg, this paper compares group, layer, instance, batch, fixed (non-aggregated) batch normalization and no normalization on CIFAR-10 with VGG-11, ResNet-18 and a LeNet-style CNN over 5000 rounds and three seeds (Section 4, Table 2).

Evidence in paper: Section 4.1; Table 2

### Group normalization performs close to layer normalization

If you use GroupNorm in FL following Hsieh et al., this paper finds GroupNorm tracks LayerNorm closely in all three architectures (e.g., 63.10% vs 63.52% on VGG-11), and notes GroupNorm as instance-based normalization may be sensitive to sample noise (Section 4.2, Table 2, Section 5).

Evidence in paper: Section 4.2; Table 2; Section 5

### Instance norm and non-aggregated BatchNorm statistics underperform

If you consider keeping BatchNorm statistics local (not aggregated) or using InstanceNorm, this paper finds fixed BatchNorm is no better than standard BatchNorm (e.g., 33.85% vs 34.38% on ResNet-18) and InstanceNorm trails LayerNorm on all three models (Table 2).

Evidence in paper: Table 2; Section 4.1

### ResNet-18 without normalization in federated learning

If you train residual networks in FL, this paper reports that ResNet-18 with no normalization reaches 60.73% versus 59.70% for LayerNorm and 34.38% for BatchNorm, attributing this to residual connections keeping weights near initialization, but notes it needs careful hyperparameter tuning for stable training (Section 4.2, Table 2).

Evidence in paper: Section 4.2; Table 2; Figure 4c

### Pathological non-IID CIFAR-10 FedAvg setup

If you replicate label-skew FL experiments, this paper uses CIFAR-10 split across 100 devices with 2 classes and 100 samples per class each, 10 clients sampled per round, 10 local epochs, batch size 64, SGD lr 0.01, 5000 rounds, on TITAN RTX GPUs (Section 4, Section 4.1).

Evidence in paper: Section 4; Section 4.1; Table 1

### Normalization effects on federated convergence speed

If you care about communication rounds to convergence in FL, this paper's accuracy-vs-rounds curves show LayerNorm converging fastest across CNN, VGG-11 and ResNet-18, with BatchNorm lagging and VGG-11 LayerNorm gaining 11.96 and 27.01 points over BatchNorm and no normalization (Section 4.2, Figure 4).

Evidence in paper: Section 4.2; Figure 4; Table 2

### Rethinking FedBN, HeteroFL static BN and GroupNorm in FL

If you survey normalization alternatives for FL, this paper positions FedBN (local BN, limited to personalized FL), GroupNorm (Hsieh et al.) and HeteroFL's static BatchNorm, arguing none explains why BatchNorm fails and offering external covariate shift as that explanation (Section 5).

Evidence in paper: Section 5

## Questions this paper answers

- Why do feature statistics diverge across clients during federated local training?
- What is the federated counterpart of internal covariate shift?
- Why does BatchNorm hurt FedAvg accuracy even beyond data heterogeneity?
- Is averaging BatchNorm running mean and variance across clients correct?
- Which normalization layer should I use for federated training of CNNs?
- Does layer norm work better than batch norm under non-IID federated learning?
- Can differences in weight norms across clients cause some clients to be ignored in model averaging?
- How does parameter norm discrepancy affect FedAvg aggregation?
- How does the scale-invariant property of normalization layers help federated aggregation?
- Why is the gradient of a scale-invariant layer orthogonal to its weights, and what does that imply for weight norm growth?
- Do two clients trained from the same initialization end up with different per-channel activation distributions?
- Is there a simple experiment showing BatchNorm statistics mismatch between federated clients?
- How do group norm, instance norm, layer norm and batch norm compare in federated learning on CIFAR-10?
- Benchmark of normalization layers for FedAvg with VGG and ResNet
- Is group normalization as good as layer normalization in federated learning?
- GroupNorm vs LayerNorm accuracy under non-IID FedAvg
- Does not aggregating BatchNorm running statistics fix BatchNorm in federated learning?
- How does instance normalization perform in federated CNN training?
- Can ResNet be trained in federated learning without any normalization layer?
- Why do residual connections help normalization-free federated training?
- What hyperparameters are used for FedAvg on CIFAR-10 with 2 classes per client?
- Example federated learning experimental setup with 100 clients and pathological non-IID partition
- Which normalization layer reduces the number of communication rounds in FedAvg?
- How much does normalization choice change VGG accuracy in non-IID federated learning?
- What are the alternatives to BatchNorm proposed for federated learning and what are their limitations?
- Is FedBN applicable to non-personalized global-model federated learning?

## Citation

```bibtex
@inproceedings{du2022fednorm,
  title     = {Rethinking Normalization Methods in Federated Learning},
  author    = {Du, Zhixu and Sun, Jingwei and Li, Ang and Chen, Pin-Yu and Zhang, Jianyi and Li, Hai and Chen, Yiran},
  booktitle = {Proceedings of the 3rd International Workshop on Distributed Machine Learning (DistributedML '22)},
  year      = {2022},
  doi       = {10.1145/3565010.3569062},
  url       = {https://arxiv.org/abs/2210.03277}
}
```

Index ID: zdu-md-84045b
