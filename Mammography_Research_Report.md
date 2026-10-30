# Multi-view Mammography Classification Using Deep Learning

*Laboratory 4 report · Project Three*

Vilnius Gediminas Technical University | Deep Learning

Dataset: VinDr-Mammo · Report date: October 2026

## Abstract

This study evaluates whether combining craniocaudal (CC) and mediolateral oblique (MLO) mammographic views improves breast-level BI-RADS suspicion classification relative to CC-only learning. VinDr-Mammo annotations were converted to a binary endpoint (BI-RADS 1–2 versus BI-RADS 4–5; BI-RADS 3 excluded). We evaluated transfer-learning baselines, two-branch concatenation and attention fusion, loss functions, image resolution and augmentation. Final single-view and multi-view ResNet50 systems were trained with three random seeds and evaluated on a held-out test cohort of 1,907 breast pairs, including 99 positive cases. Multi-view achieved higher mean test AUROC (0.8005 versus 0.7880) and average precision (0.4555 versus 0.4307). However, the paired study-cluster bootstrap 95% confidence intervals of the differences included zero (AUROC: −0.0163 to 0.0424; AUPRC: −0.0067 to 0.0555). Error analysis and Grad-CAM also revealed substantial threshold-dependent trade-offs and non-localized model responses. The results indicate a possible but uncertain incremental value of multi-view fusion; they do not demonstrate clinical diagnostic readiness.

Keywords: mammography; VinDr-Mammo; multi-view learning; ResNet50; focal loss; Grad-CAM; bootstrap.

## 1. Clinical problem and research question

Screening mammography ordinarily relies on complementary views of breast tissue. A single image can contain overlapping structures or incomplete visibility, motivating methods that integrate CC and MLO projections. The study asks: Does learning from both mammographic views produce more reliable breast-level abnormality predictions than classifying CC images independently? The experimental endpoint is suspicion according to radiologist BI-RADS categories, not pathology-confirmed cancer. Accordingly, “positive” and “negative” throughout refer to suspicion labels, not biological cancer status.

We distinguish discrimination across thresholds (AUROC and average precision, denoted AUPRC) from classification at the prespecified score threshold of 0.5 (sensitivity, specificity, precision and F1). This distinction is critical because a system may rank suspicious cases more effectively while producing more threshold-specific errors.

## 2. Dataset and cohort description

VinDr-Mammo is a public mammography dataset distributed through PhysioNet. The project uses breast-level annotations, finding-level annotations and metadata, with CC/MLO images prepared locally as PNG files from original mammograms. Each analytic row represents one breast with matched CC and MLO paths. The original image archive was too large for unrestricted full-cohort experimentation; the image availability and sample selection procedure therefore constrain generalizability.

| Partition | Breast pairs | Low suspicion (0) | Suspicious (1) | Positive rate |
| --- | --- | --- | --- | --- |
| Train | 1,352 | 1,014 | 338 | 25.0% |
| Validation | 1,150 | 1,093 | 57 | 5.0% |
| Test | 1,907 | 1,808 | 99 | 5.2% |

Training is deliberately enriched with positive breast pairs relative to validation and test. The pronounced class imbalance in the evaluation partitions makes simple accuracy a poor selection criterion. The held-out test cohort contains 998 unique study identifiers. No study identifier appeared in more than one partition; however, the available checks do not by themselves establish patient-level independence between distinct studies, an important requirement and outstanding limitation.

## 3. Data preparation and split strategy

Binary targets were formed from BI-RADS 1 and 2 (class 0) and BI-RADS 4 and 5 (class 1); category 3 was excluded to obtain a sharper low-versus-high-suspicion contrast. The initial exploratory analysis assembled breast-pair CSV files and retained them unchanged across downstream experiments. Paths were verified before final training, and both views were matched by study identifier and laterality. This breast-level grouping prevents artificial labeling of independent mammograms as unrelated observations.

For CNN input, each PNG was resized within a square canvas while preserving the aspect ratio and padding the remaining area. Images were replicated into three channels and normalized with ImageNet mean and standard deviation, matching pretrained networks. The principal experiments used 224 × 224 images; the high-resolution ablation and final comparison used 512 × 512. Strong transformations that could remove subtle findings or invert anatomical orientation were avoided.

## 4. Deep-learning methodology

### 4.1 Single-view transfer-learning baselines

ImageNet-pretrained ResNet50, DenseNet121 and EfficientNet-B0 were evaluated as independent single-view CNN architectures. A standard binary classification head produced one logit, converted to a positive-class score by sigmoid. Early model comparisons used fixed splits and a common training implementation. Their results are developmental validation estimates, not final held-out test performance.

### 4.2 Multi-view fusion

The two-branch network extracted a 2,048-dimensional representation independently from each of the CC and MLO images using ResNet50. Concatenation yielded 4,096 features, passed through a 256-unit fully connected layer, ReLU, dropout and a single-logit classifier. An attention variant instead learned view-level weights and computed a weighted sum of the two feature vectors. Attention weights measure model-internal view weighting, not verified anatomical importance.

Initial multi-view experiments froze both ImageNet encoders and trained only the fusion head. Subsequent partial fine-tuning unfroze each encoder’s final residual block (layer4). Because the initial single-view and multi-view training strategies differed, those developmental comparisons cannot isolate the causal effect of adding MLO.

### 4.3 Final training protocol

The final comparison applied common 512 × 512 preprocessing, focal loss (alpha = 0.75; gamma = 2), Adam optimizer, ten epochs, matched effective batch size 8, and three independent seeds (10, 21, 42). Only the final residual block and classifier were trainable. Encoder and head learning rates were 1×10⁻⁵ and 1×10⁻⁴, respectively. Batch size 1 with gradient accumulation and automatic mixed precision accommodated an NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB VRAM. BatchNorm running statistics were held fixed during training. The best epoch was selected by validation AUPRC; test labels played no role in checkpoint selection.

## 5. Experimental design and hypotheses

Experiments examined (i) which single-view CNN performs best, (ii) CC versus MLO representation, (iii) concatenation versus learned view attention, (iv) loss weighting for an imbalanced endpoint, (v) spatial resolution, and (vi) moderate affine augmentation. The central hypothesis was that complementary information from CC and MLO would raise average precision on held-out breasts. A secondary hypothesis was that focal loss or higher resolution would improve detection of the uncommon suspicious class. Developmental ablations were conducted in separate stages; they are not a full factorial design. Only within-stage comparisons that hold the stated remaining factors sufficiently constant can support attribution to a specific modification.

## 6. Baseline results

| Single-view validation configuration | AUROC | AUPRC | Sensitivity | F1 |
| --- | --- | --- | --- | --- |
| CC ResNet50 | 0.7316 | 0.2262 | 0.2632 | 0.2830 |
| MLO ResNet50 | 0.6725 | 0.1344 | 0.1579 | 0.1636 |

The CC-based ResNet50 provided stronger discrimination than the MLO-only variant. This result justified choosing CC as the single-view comparator and ResNet50 as the reference architecture for further fusion experiments; it does not imply that MLO is clinically redundant.

## 7. Architecture and fusion comparison

| Validation configuration | AUROC | AUPRC | Sensitivity | F1 |
| --- | --- | --- | --- | --- |
| Frozen CC+MLO concatenation | 0.7373 | 0.1532 | 0.1228 | 0.1867 |
| Partial fine-tuning CC+MLO | 0.7188 | 0.1716 | 0.1754 | 0.2198 |
| Controlled concatenation head | 0.7217 | 0.1527 | 0.1228 | 0.1667 |
| Controlled attention head | 0.6786 | 0.1226 | 0.1228 | 0.1647 |

Concatenation outperformed the tested attention head in the controlled cached-feature experiment, although the fusion heads differed in parameter count. Partial fine-tuning increased AUPRC and sensitivity relative to frozen fusion but reduced AUROC, with sharply declining training loss and fluctuating validation loss. These results motivated a simpler concatenation-based final model. They also indicate that attention alone did not guarantee improved performance.

## 8. Ablation study

### 8.1 Loss function

| Loss (224 × 224 CC) | AUROC | AUPRC | Sensitivity | F1 |
| --- | --- | --- | --- | --- |
| BCE | 0.7316 | 0.2262 | 0.2632 | 0.2830 |
| Weighted BCE (positive weight 3) | 0.7321 | 0.2394 | 0.4211 | 0.2526 |
| Focal (α 0.75, γ 2) | 0.7410 | 0.2499 | 0.4386 | 0.2793 |

Weighted BCE and focal loss increased sensitivity relative to standard BCE; focal loss achieved the best AUROC and AUPRC in this run, while BCE retained the slightly higher F1 at threshold 0.5. The choice of focal loss for final training was based on the ranking metrics and clinical preference to avoid missing positive cases, despite possible precision costs.

### 8.2 Image resolution

| Resolution | AUROC | AUPRC | Sensitivity | F1 |
| --- | --- | --- | --- | --- |
| 224 × 224 | 0.6781 | 0.1354 | 0.3509 | 0.2312 |
| 512 × 512 | 0.7858 | 0.3903 | 0.5439 | 0.2844 |

The 512 × 512 model yielded substantially higher validation average precision. A plausible mechanism is better retention of spatial details, but the physical batches differed (8 versus 2), affecting BatchNorm statistics despite an effective batch size of 8. Consequently the observed improvement cannot be attributed exclusively to resolution. High resolution also increased computation and memory requirements.

### 8.3 Augmentation

| 512 × 512 CC | AUROC | AUPRC | Sensitivity | Specificity | F1 |
| --- | --- | --- | --- | --- | --- |
| Without affine augmentation | 0.7858 | 0.3903 | 0.5439 | 0.8811 | 0.2844 |
| Moderate affine augmentation | 0.7918 | 0.2939 | 0.5789 | 0.9186 | 0.3687 |

Moderate rotations (±10°), translations (up to 5%) and scaling (0.95–1.05) improved thresholded F1, sensitivity and specificity, but lowered AUPRC. Thus improvement depended on the metric and threshold; this augmentation was not adopted in the final matched CC-versus-CC+MLO comparison.

![Figure 1](Mammography_Research_Report_assets/figure-1.png)

*Figure 1. Validation average precision for distinct ablation families; cross-family bar heights should not be interpreted as isolated causal effects.*

## 9. Held-out test performance and statistical analysis

After validation-based model selection, all six seed-specific checkpoints were evaluated once on 1,907 held-out breast pairs. The summary below reports mean and sample standard deviation across the three seeds, reflecting training variability on one fixed test set.

| Test metric | Single-view (mean ± SD) | Multi-view (mean ± SD) |
| --- | --- | --- |
| AUROC | 0.7880 ± 0.0093 | 0.8005 ± 0.0070 |
| AUPRC | 0.4307 ± 0.0279 | 0.4555 ± 0.0072 |
| Sensitivity | 0.5286 ± 0.0254 | 0.5219 ± 0.1108 |
| Specificity | 0.8969 ± 0.0083 | 0.9000 ± 0.0662 |
| F1 | 0.3100 ± 0.0013 | 0.3328 ± 0.0810 |

![Figure 2](Mammography_Research_Report_assets/figure-2.png)

*Figure 2. Mean held-out test AUROC and AUPRC for three independently trained checkpoints per architecture.*

Mean multi-view minus single-view AUPRC was +0.0248, with positive differences at seeds 10 and 42 and a negative difference at seed 21. Cluster bootstrap (1,000 resamples of 998 unique studies, keeping paired breast observations together) estimated a mean AUROC difference of +0.0125 (95% percentile CI −0.0163 to +0.0424) and mean AUPRC difference of +0.0248 (95% CI −0.0067 to +0.0555). Both intervals contain zero; the test cohort does not establish a statistically robust difference. These intervals quantify uncertainty over sampled studies conditional on the trained checkpoints; they do not incorporate all training variability or distribution shift.

## 10. Error analysis

Representative false-positive, false-negative and high-confidence errors were selected from multi-view seed 42, along with cases correctly classified by only one architecture. At threshold 0.5 the single-view confusion counts were TN = 1,636, FP = 172, FN = 49, TP = 50; the multi-view counts were TN = 1,585, FP = 223, FN = 41, TP = 58. Thus multi-view detected eight additional suspicious breasts but flagged 51 additional low-suspicion breasts. Case-level comparison found 91 errors corrected, 134 errors introduced, and 130 errors shared by both systems. This emphasizes that improved ranking can coexist with more errors at a particular threshold.

| Seed 42 test confusion counts | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| Single-view | 1,636 | 172 | 49 | 50 |
| Multi-view | 1,585 | 223 | 41 | 58 |

The most confident mistakes were selected using score assigned to the incorrect class, but uncalibrated sigmoid outputs should not be interpreted as validated confidence probabilities. Qualitative observations in the available notebooks did not include radiologist verification or systematic adjudication of all image-level explanations; therefore possible artifacts, difficult findings and label uncertainty remain hypotheses rather than established failure causes.

## 11. Explainability analysis

Gradient-weighted Class Activation Mapping (Grad-CAM) was calculated on the last convolutional block of each ResNet50 encoder in the multi-view model. Maps for CC and MLO were aligned with aspect-ratio-preserved preprocessed images. Examples of true positives, true negatives, false positives, false negatives, and cases corrected or introduced by fusion were inspected. Some heatmaps displayed localized activation within the breast, whereas others highlighted borders, background or diffuse regions. Different patterns occurred across views and between correct and incorrect predictions.

![Figure 3](Mammography_Research_Report_assets/figure-3.png)

*Figure 3. Representative Grad-CAM visualization extracted directly from the evaluation notebook (seed 42). The map explains positive-logit contribution, not lesion location.*

Grad-CAM is descriptive rather than a measure of clinical correctness: without lesion-level expert annotation, a high-intensity map cannot establish that the network has localized or recognized a malignancy. Maps based on the positive logit also should not be interpreted as complete explanations of negative decisions.

## 12. Limitations and threats to validity

Important limitations include: (1) verified study-level but not conclusively verified patient-level independence across partitions; (2) selected images available locally rather than an unrestricted full-dataset benchmark, producing possible cohort and selection bias; (3) a strongly enriched training positive-class fraction relative to evaluation partitions; (4) BI-RADS grouping instead of pathology-proven cancer ground truth, with BI-RADS 3 excluded; (5) limited numbers of positive validation (57) and test (99) breasts; (6) one institutionally sourced public dataset without external validation; (7) early baseline and ablation comparisons with some changing training conditions; (8) partial fine-tuning and greater model capacity in multi-view; and (9) image-derived heatmap interpretations lacking lesion-level independent clinical confirmation.

The final three-seed protocol improves reproducibility relative to a single trial, but is not a substitute for external validation or larger patient-level uncertainty studies. The bootstrap percentile intervals preserve study clusters but are conditional on the observed cohort and selected model family. Threshold 0.5 was kept consistent instead of being optimized after seeing test performance.

## 13. Clinical implications

A modest increase in test average precision suggests CC+MLO may supply complementary ranking information, but the large false-positive cost observed for seed 42 and the overlapping uncertainty intervals prevent claims of clinical benefit. The models should be understood as research classifiers for a simplified suspicion task, not autonomous screening devices. Before any practical deployment, a patient-level independently curated dataset, external multi-site testing, lesion-level validation, subgroup analysis, probability calibration, prespecified referral thresholds and a reader study would be necessary.

## 14. Conclusion

This study implemented and evaluated a reproducible VinDr-Mammo breast-level binary classification pipeline across single-view models, multi-view fusion methods and controlled ablations. The selected two-branch ResNet50 concatenation model achieved higher mean test AUROC (0.8005) and AUPRC (0.4555) than the CC-only ResNet50 (0.7880 and 0.4307) across three seeds, yet bootstrap intervals included no difference. Higher input resolution gave the largest developmental AUPRC change; focal loss raised positive-class sensitivity; attention fusion did not consistently improve over concatenation. Error and Grad-CAM analyses highlighted both the potential and limitations of multi-view evidence. Multi-view learning may improve suspicion ranking, but its superiority and clinical value remain unproven.

## References

[1] Vilnius Gediminas Technical University. Deep Learning Laboratory Four: Medical Image Analysis Using Real-World Datasets. Course FMISB21502, supplied assignment.

[2] Nguyen H. T. et al. VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography. Scientific Data (2023). Dataset distributed via PhysioNet: https://physionet.org/content/vindr-mammo/1.0.0/

[3] He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition. CVPR (2016).

[4] Lin T.-Y. et al. Focal Loss for Dense Object Detection. ICCV (2017).

[5] Selvaraju R. R. et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV (2017).

[6] Efron B., Tibshirani R. J. An Introduction to the Bootstrap. Chapman & Hall/CRC (1993).
