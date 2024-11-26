---
layout: distill
title: Reassessing adversarial training with fixed data augmentation
description: A recent bug discovery on Pytorch+Numpy got me thinking- how much does this bug impact adversarial robustness?
tags:
- robustness
- pytorch bug
- adversarial training
date: 2021-06-24
thumbnail: assets/img/adv_rob_aug.jpg
citation: true
featured: false
pretty_table: true
categories: exploration


authors:
  - name: Anshuman Suri
    url: "https://anshumansuri.com/"
    affiliations:
      name: University of Virginia

bibliography: combined.bib

toc:
  - name: Overview
  - name: Experiments
  - subsections:
    - name: Standard Training
    - name: Adversarial Training
  - name: Takeaway
---

## Overview

A couple months ago, a [post on Reddit](https://www.reddit.com/r/MachineLearning/comments/mocpgj/p_using_pytorch_numpy_a_bug_that_plagues/) highlighted a bug in PyTorch + NumPy that affects how data augmentation works (see image above). Knowing nearly all of my projects use this combination, I read through the [linked blog](https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/) by Tanel Pärnamaa to see what it was all about. I was a bit shocked that it took our community this long to notice a bug this severe! Nearly all data-loaders use more than one worker. Unfortunately, not many people (clearly, since it took us all so long to notice this bug) sit down to debug data augmentation at this level within their ML pipeline.

Reading through this bug, I remembered how (proper) data-augmentation had been proposed as a means to reduce robust overfitting by authors at DeepMind <d-cite key="rebuffi2021fixing"> </d-cite>. This paper got me thinking: "Could fixing this augmentation bug and rerunning adversarial training lead to gains in robustness?". Curious to see the impact of fixing this data augmentation bug, I decided to run some experiments of my own. You can head over to [the repository](https://github.com/iamgroot42/aug_robust_blogpost) and run them yourself if you want.

## Experiments

I chose the CIFAR-10 dataset: small enough to iterate experiments fast and challenging enough to observe performance gains.

### Standard Training

Interestingly, standard training with the fixed data-augmentation pipeline **hurt** performance a bit, compared to using faulty augmentation:

| Model | Standard Accuracy (%) | Robust Accuracy (ε = 8/255) (%) |
| ----------- | ----------- | ----------- |
| Standard | 89.140 | 0.000 |
| Standard (augmentation) | 94.720 | 0.000 |
| Standard (fixed augmentation) | 94.620 | 0.000 |

### Adversarial Training

Not thinking much about the 0.1% performance drop (probably statistical noise, right?), I ran adversarial training with $$L_\infty$$ robustness ($$\epsilon=\frac{8}{255}$$):

| Model | Standard Accuracy (%) | Robust Accuracy (ε = 8/255) (%) | Robust Accuracy (ε = 16/255) (%) |
| ----------- | ----------- | ----------- | ----------- |
| Robust | 79.520 | 44.370 | 15.680 |
| Robust (augmentation) | 86.320 | 51.400 | 17.480 |
| Robust (fixed augmentation) | 86.730 | 51.880 | 17.570 |

As visible here, there's an absolute 0.4% performance gain for $\epsilon=\frac{8}{255}$, and 0.09% performance gain for $\epsilon=\frac{4}{255}$, when using the fixed augmentation pipeline. Although the 0.09% here is not very significant, the 0.4% improvement seems non-trivial. This improvement is especially significant compared to the kind of performance differences reported on [benchmarks](https://robustbench.github.io/#div_cifar10_Linf_heading) for this dataset. Additionally, accuracy on clean data sees an improvement as well: absolute 0.41% change.

Not wanting to make any claims based on experiments on just the $L_\infty$ norm, I reran the same set of experiments for the $$L_2$$ norm ($$\epsilon=1$$).

| Model | Standard Accuracy (%) | Robust Accuracy (%), ε = 0.5 | Robust Accuracy (%), ε = 1 |
| ----------- | ----------- | ----------- | ----------- |
| Robust | 78.190 | 61.740 | 42.830 |
| Robust (augmentation) | 80.560 | 67.200 | 51.140 |
| Robust (fixed augmentation) | 81.070 | 67.620 | 51.220 |

Performance gains appear in this case as well. Accuracy on clean data bumps up by 0.51%, while robustness on $$\epsilon=0.5$$ and $$\epsilon=1.0$$ improves by 0.42% and 0.08%, respectively. The fact that this case sees a consistent, albeit small, improvement in both clean and perturbed-data performance hints at how simply fixing this augmentation may provide a nice bump in existing training methods. It is very much possible that these gains are just coincidental fluctuations in the randomness of model training. Regardless, fixing data-loaders is something that should be done anyway. The goal of these experiments was to try and quantify the impact of improper augmentation. It would be great if someone with sufficient resources could run these experiments on a larger scale to rule out statistical noise.

## Takeaway

Fixing data augmentation can have a non-trivial (and positive) impact when training for robustness. Anyone training robust models (especially with adversarial training, since that is what I tested on) should fix their data-loaders.
