# Sky Segmentation Project -- Architecture & Strategy Discussion Summary

## 1. Project Goal Clarification

Current objective is **not cross-camera generalization research**, but:

> Maximize segmentation performance across the known camera set and
> ensure all defined scenarios are predicted well.

Deployment assumption: - Model will operate on these existing cameras. -
Therefore, in-domain performance is primary KPI.

------------------------------------------------------------------------

# 2. Data Split Strategy (Final Decision: Scheme S)

## Why not strict camera-disjoint?

Camera-disjoint split measures domain generalization. However, our goal
is not testing unseen cameras but maximizing performance on current
cameras.

Therefore:

## Final Split Strategy

For **each camera independently**:

-   80% → Train
-   10% → Validation
-   10% → Test
-   Split by timestamp or filename order (deterministic rule)

### Rationale

-   Every camera contributes to training.
-   Every camera also contributes to final evaluation.
-   No identical frame appears across splits.
-   Test set is locked and shared across all pipelines for fair
    comparison.

------------------------------------------------------------------------

# 3. Model Architecture Understanding

## Encoder

Encoder extracts abstract feature representation from image.

Common encoders: - CNN (ResNet, UNet encoder, etc.) - Vision Transformer
(ViT) - DINO (self-supervised trained ViT)

DINO is NOT a different architecture --- it is a ViT trained using
self-supervised learning.

------------------------------------------------------------------------

## Decoder

Decoder transforms feature maps into task-specific outputs.

Examples: - Segmentation → upsampling + conv layers - UNet decoder -
Transformer decoder (in other tasks)

Not all tasks require a decoder: - Classification → encoder + head only

------------------------------------------------------------------------

## Head

Head is task-specific module on top of backbone.

Examples: - Classification head: Linear → Softmax - Segmentation head:
Conv + Upsample - Detection head: Bounding box regression +
classification

Head is trainable and depends on task.

------------------------------------------------------------------------

# 4. Three Pipelines to Compare

## 1) UNet (CNN-based)

Image → CNN encoder → UNet decoder → Mask

-   Task-specific supervised learning
-   Optimized for sky segmentation
-   Baseline DL model

------------------------------------------------------------------------

## 2) VLM → SAM

Image → VLM (text alignment) → Prompt → SAM → Mask

-   Uses semantic alignment
-   No segmentation training required
-   Strong zero-shot and structural ability

------------------------------------------------------------------------

## 3) DINO → DL (Optional Advanced Pipeline)

Image → DINO (frozen encoder) → Segmentation head → Mask

-   Uses self-supervised learned features
-   Train only head with sky labels
-   Hybrid between representation learning and task learning

------------------------------------------------------------------------

# 5. Evaluation Principles

All pipelines must:

-   Use identical `test_list.txt`
-   Report:
    -   Overall IoU
    -   Per-camera IoU
    -   FP / FN
    -   No-sky FP rate
-   No test set modification after locking

Optional: - 10870 as low-light diagnostic subset

------------------------------------------------------------------------

# 6. Conceptual Clarifications from Discussion

### Feed-forward vs FC

-   FC (Linear layer): single matrix transform
-   Feed-forward network (Transformer FFN): Linear → Nonlinearity →
    Linear
-   Head can contain FFN or conv layers depending on task

------------------------------------------------------------------------

### Patch Embedding & Heatmap

VLM heatmap comes from:

cos(patch_embedding_i, text_embedding_sky)

Patch similarity reshaped to spatial grid, then interpolated to pixel
resolution.

Model never receives pixel-level supervision in VLM alignment training.

------------------------------------------------------------------------

# 7. Final Direction

We prioritize:

1.  Clean data split (Scheme S)
2.  Retrain UNet baseline
3.  Run VLM → SAM on same test set
4.  Compare results
5.  Optionally implement DINO → DL pipeline

------------------------------------------------------------------------

# 8. Immediate Next Steps

## Step 1 -- Execute Data Split

-   Implement deterministic 80/10/10 per camera
-   Generate fixed train/val/test lists

## Step 2 -- Retrain UNet

-   Use new train split
-   Early stopping on val
-   Save best checkpoint

## Step 3 -- Run Evaluation

-   Evaluate UNet on locked test
-   Evaluate VLM → SAM on same test
-   Produce comparison report

## Step 4 -- Analyze Weak Scenarios

-   Identify which cameras or scenarios still underperform
-   Decide if architectural change or data augmentation needed

------------------------------------------------------------------------

# 9. Long-Term Extension

If needed: - Implement DINO backbone + segmentation head - Compare
stability and generalization vs UNet - Possibly use VLM/SAM to generate
pseudo-labels for further training

------------------------------------------------------------------------

# Final Reflection

This discussion clarified the difference between:

-   Engineering objective (maximize known domain performance)
-   Research objective (test generalization)

Current decision aligns with deployment-oriented engineering mindset.
