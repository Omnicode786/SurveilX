# Prior art and novelty audit

This is an engineering literature map, based on live primary-source checks during implementation (2026-09-27). It is not a paper manuscript or an exhaustive 2017–2026 systematic review. Abstract-level checks identify relevant methods; performance claims from unrelated benchmarks are not transferred to SurveilX. Full reproduction and recent-work saturation remain open research tasks.

## Primary-source map

| Area | Primary work | Method / relevance | Limitation for this system |
|---|---|---|---|
| Adaptive inference | [NoScope (2017)](https://people.eecs.berkeley.edu/~matei/papers/2017/vldb_noscope.pdf) | Specialized cascades reduce expensive video queries | Binary, fixed-camera queries do not establish multi-domain event understanding |
| Adaptive inference | [Chameleon (2018)](https://doi.org/10.1145/3230543.3230574) | Profiles and adapts resolution, frame rate and model configuration | Adaptation already exists; semantic risk utility needs separate validation |
| Edge filtering | [Reducto (2020)](https://www.cs.princeton.edu/~ravian/publications/reducto_sigcomm20.pdf) | Cheap frame differences select frames for inference | Filtering can hide slow or stationary events; keep service deadlines |
| Global scheduling | [VideoStorm (2017)](https://www.microsoft.com/en-us/research/publication/live-video-analytics-scale-approximation-delay-tolerance/) | Resource-quality profiles and quality/lag scheduling | Cluster evaluation does not prove single-PC or thermal behavior |
| Calibration | [On Calibration of Modern Neural Networks (2017)](https://arxiv.org/abs/1706.04599) | Held-out temperature scaling minimizes negative log likelihood | Calibration is distribution-dependent and cannot repair a bad classifier |
| Risk control | [Conformal Risk Control](https://arxiv.org/abs/2208.02814) | Controls expected monotone loss using held-out calibration | Assumptions must be checked under temporal dependence and domain shift |
| Epistemic uncertainty | [Deep Ensembles (2017)](https://arxiv.org/abs/1612.01474) | Independently trained predictors provide disagreement estimates | Extra inference/training cost must be budgeted |
| Abstention | [SelectiveNet (2019)](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf) | Learns a reject option with selective risk/coverage objectives | Coverage is not automatically a critical-event guarantee |
| Calibration interplay | [Temperature Scaling and Conformal Prediction (2024)](https://arxiv.org/abs/2402.05806) | Studies calibration and conformal set efficiency together | A better calibrated classifier can still produce less efficient sets |
| Calibration training | [On Mixup Training (2019)](https://arxiv.org/abs/1905.11001) | Mixup changes calibration and uncertainty behavior | Synthetic interpolation needs domain-specific evaluation |
| Video | [SlowFast (2019)](https://arxiv.org/abs/1812.03982) | Separate temporal sampling pathways for appearance and motion | General action recognition is a comparator, not SVA-Net novelty |
| Video pretraining | [VideoMAE (2022)](https://arxiv.org/abs/2203.12602) | Masked video reconstruction learns visual representations | Pretraining and downstream entity/interaction labels differ |
| Video | [Video Swin (2021)](https://arxiv.org/abs/2106.13230) | Local spatiotemporal windows and hierarchical representations | Cost and surveillance transfer must be measured |
| Spatial vision | [Swin Transformer (2021)](https://arxiv.org/abs/2103.14030) | Hierarchical shifted-window attention | Spatial backbone alone is not temporal surveillance reasoning |
| Entity detection | [DETR (2020)](https://arxiv.org/abs/2005.12872) | Set prediction with bipartite matching | Query-based entities and matching are established ideas |
| Entity detection | [Objects as Points (2019)](https://arxiv.org/abs/1904.07850) | Center-based object representation | Localization alone does not estimate event risk |
| Sets | [Deep Sets (2017)](https://arxiv.org/abs/1703.06114) | Permutation-invariant set aggregation | Pooling may discard interaction and trajectory structure |
| Dense detection | [Focal Loss (2017)](https://arxiv.org/abs/1708.02002) | Reweights easy examples in dense detection | Loss choice cannot substitute for new detector design |
| Tracking | [ByteTrack (2022)](https://arxiv.org/abs/2110.06864) | Associates detections across confidence ranges | Needs appropriate detection quality; not a risk classifier |
| Tracking | [Deep SORT (2017)](https://arxiv.org/abs/1703.07402) | Appearance-assisted association | Learned appearance introduces cost and domain dependence |
| Surveillance events | [Real-world Anomaly Detection in Surveillance Videos (2018)](https://arxiv.org/abs/1801.04264) | Weakly supervised anomaly learning from surveillance videos | Video-level labels cannot train trustworthy object/interaction labels by themselves |
| Multi-camera | [CityFlow (2019)](https://arxiv.org/abs/1903.09254) | Multi-camera vehicle tracking benchmark | Vehicle domain does not cover office/retail behavior |
| Constrained learning | [Constrained Policy Optimization (2017)](https://arxiv.org/abs/1705.10528) | Policy search subject to constraints | Requires environment/rollouts and assumptions absent at cold start |
| Continual learning | [Learning without Forgetting (2016/2017)](https://arxiv.org/abs/1606.09282) | Distillation preserves earlier task behavior | Retention is empirical; evaluate old domains after adaptation |
| Continual learning | [Overcoming Catastrophic Forgetting (2017)](https://arxiv.org/abs/1612.00796) | Parameter regularization preserves important prior knowledge | Regularization does not guarantee non-forgetting |
| Continual learning | [Gradient Episodic Memory (2017)](https://arxiv.org/abs/1706.08840) | Replay constraints protect previous tasks | Storage, representativeness and label quality matter |
| Human feedback | [Deep RL from Human Preferences (2017)](https://arxiv.org/abs/1706.03741) | Human comparisons supervise a learned objective | Preference learning is distinct from surveillance ground truth |
| Risk training | [Conformal Risk Training (2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/6559542f75b4452ebaaf82094c7defb7-Paper-Conference.pdf) | Couples training with risk-oriented conformal objectives | Newer alternative to examine before making any risk-control novelty claim |

## Closest-prior-art assessment

| Proposed component | Closest work | Existing mechanism | Gap to test | SurveilX difference | Novel? |
|---|---|---|---|---|---|
| RAIC | Chameleon, VideoStorm | Cost/quality configuration control | Context-dependent avoided event loss | Joint operational context and hardware governor | No established novelty; cold-start heuristic |
| ASIE | Domain conditioning and scene specialization | Domain-specific perception configuration | Profile errors and downstream degradation | Versioned manual profiles first | Engineering, not novel |
| YOLO-RAI | Conditional feature modulation, adaptive inference | Context-conditioned features and scale choice | Useful risk-zone conditioning without detector harm | Experimental identity-initialized feature adapter | Hypothesis only; not integrated/trained detector yet |
| SVA-Net | SlowFast, Deep Sets, DETR, Video Swin | Temporal features, entities, interactions | Entity persistence plus domain context under edge budgets | Entity sampling, explicit trajectories, relation messages | Custom composition; novelty unproven |
| MCRS | VideoStorm | Global resource allocation | Coverage under simultaneous incidents | Deadline-first allocation with explicit infeasibility | No |
| Arbitration | Ensembles, SelectiveNet | Disagreement and abstention | Mismatched semantic tasks/domains | Compatibility checks before fusion | No |
| Temporal reasoning | SlowFast, Video Swin | Learned temporal representations | Sparse irregular entity histories | Internal SVA temporal convolution | No independent temporal model |
| Adaptation | LwF, EWC, GEM | Regularization, distillation, replay | Noisy feedback and unseen deployment domains | Reviewed labels and candidate-only training | No |
| Human loop | Human preference learning / selective prediction | Human supervision and rejection | Operator workload and label reliability | Separate reviewer plus audit | Engineering |
| Edge scheduling | Chameleon, Reducto | Adaptive sampling/configuration | Unknown PCs and optional accelerators | Measured runtime capability and hysteresis | Engineering |
| Risk estimation | Conformal risk control | Calibrated expected loss control | Temporal dependence and rare events | Operational severity separated from probability | No calibrated threat model yet |

There are no genuine novelty claims to defend at this stage. Renaming components does not change that. The executable candidate tests an architectural hypothesis; real-domain comparisons and stronger prior-art searches are prerequisites for a contribution claim.

## Formulation selection

POMDPs express hidden scene state but require a defensible observation/transition model. CMDPs express sequential resource constraints but require policy training and validated transition costs. Contextual bandits need logged rewards and careful counterfactual evaluation. None is justified by an empty deployment with no labeled traces. A constrained resource allocator is inspectable and supports measured feasibility immediately. Expected avoided event loss should later replace the cold-start priority ordering after offline replay validation. CVaR or conformal constraints must not be added as decorative equations.

## Official implementation references

- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html): device/runtime checks and synchronization for timings.
- [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/): accelerator-specific availability and graph placement; CPU fallback must be visible.
- [AMD/Xilinx Vitis AI](https://github.com/Xilinx/Vitis-AI): hardware-specific inference stack, not a universal FPGA training backend.
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/): startup/shutdown lifecycle for the runtime.
- [Ultralytics prediction](https://docs.ultralytics.com/modes/predict/): supplied checkpoint integration, device and image-size selection.
