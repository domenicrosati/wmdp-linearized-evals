---
pretty_name: WMDP Linearized Red-Team Harness
license: apache-2.0
task_categories:
  - text-classification
tags:
  - wmdp
  - red-teaming
  - ntk
  - linearized-training
---

# WMDP linearized red-team harness

Minimal, auditable code for three WMDP-Bio measurements:

1. native full-answer inference;
2. closed-form ridge regression on frozen answer-token features;
3. cross-entropy gradient descent in the same fixed feature space.

## Important scope

**This is a red-team diagnostic, not a security certificate.**

A low endpoint only means that the declared frozen answer-token feature family
did not recover the task under the declared label budget and hyperparameter
grid. It does **not** imply resistance to:

- LoRA or full fine-tuning;
- feature learning, where the Jacobian changes during training;
- a sidecar using another encoder or raw text features;
- a different layer, prompt, scorer, optimizer, or larger label budget.

In the associated experiments, fixed-feature resistance was bypassed by an
external lexical-plus-score sidecar, and release-time tangent obstruction did
not generally survive all-layer feature learning. Use this repository to find
and report vulnerabilities, not to certify that a model is safe.

## Models

- [Synthetic DI](https://huggingface.co/domenicrosati/synthetic-di) - may
  require access authorization and may not yet contain the audited V7 package.
  A local checkpoint path can be supplied instead.
- [DI Strong-Combined](https://huggingface.co/EleutherAI/deep-ignorance-e2e-strong-filter-cb-lat)
  at revision 08ff131a561aa0863cedd7d956f974fc0cf38676.
- [Unfiltered control](https://huggingface.co/EleutherAI/deep-ignorance-unfiltered)
  at revision c8df368ff247cb90b62e21e1689260701b3ff25a.

The CLI accepts any compatible causal language model. Comparisons are matched
only when dataset, split, scorer, dtype, feature layer, rank, and selection
procedure are identical.

## Published matched results

Held-out cells report accuracy and cross-entropy on 215 sealed questions.
Higher accuracy and lower cross-entropy indicate a more successful attack.

| Model | Native | Ridge | Fixed-J CE-GD |
| --- | ---: | ---: | ---: |
| Unfiltered control | 47.44% / 1.3619 | 53.02% / 1.1944 | 53.02% / 1.1434 |
| Strong-Combined | 28.84% / 1.6669 | 26.98% / 1.5127 | 38.14% / 1.4734 |
| Synthetic DI | pending matched rerun | pending matched rerun | pending matched rerun |

The only available Synthetic-DI linearized result used a different
four-letter output-head interface: 48.33% to 49.91%, with a 0.0080 CE
reduction. It is included in results/published-results.json as a legacy,
incompatible result and is **not** mixed into the matched table.

## Install

    git clone https://github.com/domenicrosati/wmdp-linearized-evals
    cd wmdp-linearized-evals
    python -m venv .venv
    source .venv/bin/activate
    pip install -e .

The extraction path needs a CUDA GPU and downloads WMDP-Bio plus the selected
model from Hugging Face. Authenticate first if the model is private:

    hf auth login

## Run

Unfiltered positive control:

    wmdp-linearized all \
      --preset unfiltered \
      --output-dir runs/unfiltered

Strong-Combined:

    wmdp-linearized all \
      --preset strong-combined \
      --output-dir runs/strong-combined

Synthetic DI from Hugging Face:

    wmdp-linearized all \
      --preset synthetic-di \
      --output-dir runs/synthetic-di \
      --trust-remote-code

Synthetic DI from a local checkpoint:

    wmdp-linearized all \
      --model /path/to/checkpoint \
      --model-label synthetic-di-local \
      --output-dir runs/synthetic-di-local \
      --trust-remote-code

To separate costly GPU extraction from cheap CPU attacks:

    wmdp-linearized extract \
      --preset strong-combined \
      --cache runs/strong-combined/cache.pt

    wmdp-linearized attack \
      --cache runs/strong-combined/cache.pt \
      --output runs/strong-combined/results.json

    wmdp-linearized audit \
      --cache runs/strong-combined/cache.pt \
      --results runs/strong-combined/results.json

## Frozen protocol

- Dataset: cais/wmdp, configuration wmdp-bio, test split.
- Interface: four full answer strings scored by mean answer-token log
  probability.
- Seed: 17.
- Split: 689 construction, 172 validation, 215 sealed reporting questions.
- Features: final hidden state averaged over each candidate's answer-token
  prediction positions, then centered across the four candidates.
- Basis: label-free rank-512 randomized SVD fitted on all 861 non-report
  questions.
- Scaling: root-mean-square scaling fitted on construction questions only.
- Selection: ridge strength or GD learning rate/checkpoint selected by
  validation cross-entropy only.
- Reporting labels are never used for fitting or selection.

The results file stores exact split indices, grids, selected cells, hashes, and
trajectories needed for replay.

## What is proved

Let H be the fixed feature matrix, Y the centered target correction, W the
readout, and lambda a nonnegative ridge coefficient. Ridge solves

    minimize_W ||H W - Y||_F^2 + lambda ||W||_F^2.

Its saved solution is the exact optimum of that regularized squared-loss
problem, up to numerical linear-solve error.

Fixed-J CE-GD predicts

    z(x) = z_0(x) + Phi(x) w

and optimizes cross-entropy over w while Phi remains fixed. Cross-entropy is
convex in w, so standard gradient descent converges under the usual smoothness
and step-size conditions.

Neither proof covers held-out generalization or feature learning. LoRA and
full fine-tuning update the model itself, so the Jacobian J_t and kernel
K_t = J_t J_t^T change during training.

## Responsible reporting

Report native, ridge, and CE-GD accuracy and cross-entropy. Include:

- model revision and tokenizer;
- cache and result SHA-256 hashes;
- split sizes and seed;
- feature rank and extraction dtype;
- complete selection grids;
- a positive control showing the attack learns on an undefended model;
- the limitation that fixed-feature results do not cover feature learning.

Do not interpret WMDP performance as a complete measurement of biological
capability or operational risk.

## Development

    pip install -e '.[dev]'
    pytest

## License

Apache-2.0. Model and dataset licenses remain their respective owners' terms.
