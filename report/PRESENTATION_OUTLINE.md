# Presentation outline — Casting defect inspection (ViT vs ResNet)

**Target length:** 15–20 minutes + Q&A  
**Format:** ~18 slides

---

## Slide 1 — Title

- Title, your name, program, supervisor, date
- Subtitle: data-efficient comparison on industrial casting images

## Slide 2 — Motivation

- Manual inspection bottlenecks in foundries
- Cost of missed defects vs false alarms
- Photo: top-view impeller (from dataset)

## Slide 3 — Research questions

- ViT vs ResNet under limited labels?
- Accuracy vs latency vs model size?
- Explainability for trust?

## Slide 4 — Dataset

- Kaggle Pilot Technocast casting dataset
- Binary classes: `ok_front` / `def_front`
- Prefer ~1,300 non-augmented images; mention controlled lighting
- Class counts bar chart `[INSERT FIGURE]`

## Slide 5 — Preprocessing & leakage control

- Resize 224, grayscale→RGB, ImageNet norm
- Stratified splits; group-wise split for augmented filenames
- Fixed val/test; varying train fractions 25/50/100%

## Slide 6 — Architectures

- Side-by-side: ResNet-18 (local conv) vs ViT-B/16 (patches + attention)
- Pretraining sources: ImageNet / IN21K

## Slide 7 — Training protocol

- AdamW, early stopping on val macro-F1
- Seed 42, batch 32, max 30 epochs
- Same protocol both models

## Slide 8 — Metrics

- Accuracy, macro-F1, defect P/R
- Confusion matrix, train time, latency, MB

## Slide 9 — Results overview

- Table or grouped bar: macro-F1 vs train fraction `[INSERT RESULT]`
- Highlight best model per regime

## Slide 10 — ResNet curves

- Training/val loss and F1 `[INSERT FIGURE]`

## Slide 11 — ViT curves

- Same for ViT `[INSERT FIGURE]`

## Slide 12 — Confusion matrices

- 2×2 grids for six runs or best/worst cases `[INSERT FIGURE]`

## Slide 13 — Interpretability

- Grad-CAM on false negatives/positives
- ViT attention overlay on defect regions

## Slide 14 — Efficiency & deployment

- Latency and model size comparison chart
- Edge deployment considerations (Pi-class hardware mention optional)

## Slide 15 — UAE / Operation 300bn

- Advanced manufacturing & AI adoption in UAE industrial strategy
- Data-efficient AI lowers barrier for regional SMEs

## Slide 16 — Limitations

- Single product view, binary labels, no severity taxonomy
- Domain shift if lighting changes

## Slide 17 — Conclusions

- Answer RQ1–RQ3 with numbers `[INSERT RESULT]`
- Practical recommendation for foundry partner

## Slide 18 — Thank you / Q&A

- GitHub + Colab links
- Contact email

---

## Speaker notes (optional)

- Slide 5: emphasize why grouping `cast_*_*` filenames matters for augmented Kaggle zip.
- Slide 13: prepare one **missed defect** and one **false alarm** story for Q&A.
