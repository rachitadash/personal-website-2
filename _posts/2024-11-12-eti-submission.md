---
layout: distill
title: My submission to the ETI Challenge
description: Description of my entry to the ETI (Erasing the Invisible) challenge (co-located with NeurIPS) for watermark-removal.
tags:
- watermark removal
- robustness
- adversarial examples
- diffusion models
date: 2024-11-12
thumbnail: assets/img/the_great_wave_off.jpg
citation: true
featured: false
categories: competition

authors:
  - name: Anshuman Suri
    url: "https://anshumansuri.com/"
    affiliations:
      name: Northeastern University

bibliography: combined.bib

toc:
  - name: Adversarial Rinsing
  - subsections:
    - name: Generating Adversarial Perturbations
    - name: Augmentations for Robustness
    - name: Generative Models
  - name: What didn’t work?
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Takeaways
---

This post describes an approach developed for the [Erasing the Invisible](https://erasinginvisible.github.io/) challenge at NeurIPS 2024.
My method combined “rinsing” with adversarial techniques, designed for both the black-box and beige-box competition tracks. 
Although my solution didn’t secure a top spot, I saw potential in the methodology and wanted to document it to possibly aid future research and development in this area.

## Adversarial Rinsing

The central idea behind my approach is blending "rinsing"<d-cite key="an2024waves"></d-cite> with adversarial perturbations. "Rinsing" here means passing an image through a diffusion model multiple times, intending to erode watermarks present in the input. For adversarial examples, I used the SMI$^2$FGSM<d-cite key="wang2022enhancing"></d-cite> attack because of its success with transfer-based attacks<d-cite key="suya2024sok"></d-cite>.
The objective of these adversarial perturbations is to disrupt the latent space representation of the image, aiming to dislodge any potential latent-space watermarks.

### Generating Adversarial Perturbations

I used a joint loss that maximizes the separation of the perturbed image’s embedding from the original in two ways:

- **Embedding Space Distance**: A loss that combines norm-distance and cosine-distance for better embedding separation.

```python
class MSEandCosine(ch.nn.Module):
  ...
    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        # Flatten to compute cosine similarity
        csn_loss = 1 - self.csn(output.view(output.size(0), -1), target.view(target.size(0), -1))

        # Combined Loss
        loss = (1 - self.alpha) * mse_loss + self.alpha * csn_loss
        return loss
```

- **Image Quality Loss**: This component uses differentiable metrics like PSNR, SSIM, LPIPS, aesthetics, and artifacts scores. By optimizing this loss, my aim was to preserve the original image quality while removing the watermark.

```python
class NormalizedImageQuality(ch.nn.Module):
    ...

    def forward(self, output, target):
        """
            Output here is generated image, target is original image
        """
        outputs_aesthetics, outputs_artifacts = self._compute_aesthetics_and_artifacts_scores(output)
        final_score = -4.5e-2 * outputs_aesthetics + 1.44e-1 * outputs_artifacts
        return final_score

        lpips_score = self._lpips(output, target)
        psnr_score = self._psnr(output, target)
        ssim_score = self._ssim(output, target)
        outputs_aesthetics, outputs_artifacts = self._compute_aesthetics_and_artifacts_scores(output)

        # Differentiable NMI is too slow, so ignoring it for now
        # nmi_score = differentiable_nmi(output, target)

        if self.target_aesthetics is None:
            self.target_aesthetics, self.target_artifacts = self._compute_aesthetics_and_artifacts_scores(target)

        delta_aesthetics = outputs_aesthetics - self.target_aesthetics
        delta_artifacts = outputs_artifacts - self.target_artifacts

        # Differentiable metrics!
        weighted_scores = [
            (psnr_score, -2.22e-3), # PSNR
            (ssim_score, -1.13e-1), # SSIM
            (lpips_score, 3.41e-1), # LPIPS
            (delta_aesthetics, 4.5e-2), # Delta-Aesthetics
            (delta_artifacts, -1.44e-1), # Delta-Artifacts
        ]

        # Aggregate weighted scores
        final_score = sum([score * weight for score, weight in weighted_scores])

        # Want to be close to zero
        return ch.abs(final_score)
```

The optimization then proceeds with a series of augmentations (described below), with an $L_\infty$ norm constraint on the added perturbation.
During experimentation, I also tried minimizing detection by watermark-detection models like WAVES but found it degraded performance.<d-footnote>I didn't play around too much with the hyper-parameters in my algorithm. Maybe some hyper-parameter tuning could help?</d-footnote>
```python
...
"stable_sig": (ClassificationWrapper(stable_sig), "classify"),
"tree_ring": (ClassificationWrapper(tree_ring), "classify"),
"stegastamp": (ClassificationWrapper(stegastamp), "classify")
```

### Augmentations for Robustness

To enhance the effectiveness of the attack, a diverse set of differentiable augmentations are integrated into SMI$^2$FGSM. These augmentations are chosen to closely match the kind of augmentations usually used in watermark-insertion algorithms: `Random crop,  Gaussian blur, Gaussian noise, JPEG compression, Noise in the FFT domain, Rotation, Motion Blur, Random brightness, Random contrast, Random hue, Horizontal flips`. I additionally used Mixup using a set of clean images. To avoid the attack overfitting to a specific augmentation, I randomly sampled from the set of possible augmentations at each iteration.

```python
  transformation_functions = [
        random_crop,
        gaussian_blur,
        ...
        mixup
    ]
    # Randomly pick one of the transformation functions
    random_transform = transformation_functions[np.random.randint(0, len(transformation_functions))]
```

I also sample the hyper-parameters for each of these augmentations from a wide range of values to avoid potential overfitting.

```python
def motion_blur(x):
  angle = np.random.randint(5, 175)
  direction = np.random.choice([-1, 1])
  return kornia.filters.motion_blur(x, kernel_size=15, direction=direction, angle=angle, border_type='constant')
```

### Generative Models

Empirical observations during implementation revealed that the `waves` and `openai/consistency-decoder` generative models yielded the best results. Flipping their order or adding another diffusion/generative models only made the final image worse, since multiple rinsing runs were presumably degrading image quality.

```python
def get_class_scaled_logits(model, features, labels):
    outputs = model(features).detach().cpu().numpy()
    num_classes = np.arange(outputs.shape[1])
    values = []
    for i, output in enumerate(outputs):
        label = labels[i].item()
        wanted = output[label]
        not_wanted = output[np.delete(num_classes, label)]
        values.append(wanted - np.max(not_wanted))
    return np.array(values)
```

## What didn't work?

A ton of things! I experimented with a lot of compression algorithms, adding noise to images, and various combinations of all of these methods. I also tried adding adversarial perturbations generated using some Imagenet classifier (as a proxy for perturbations that could shift the latent space in favor of avoiding watermark detection). None of them worked, with most of them retaining most of their watermarks. To be honest this did surprise me a bit- stepping into this field I did not realize that these image watermarks could be so robust. I also tried my adversarial-rinse approach, but without "rinsing" - using watermark-detection models as my target models, with varying number of iterations. While that does work to some extent, its performance is nowhere close to that when rinsing is introduced. While the converse is also true, rinsing by itself proved to be much more useful than only adversarial perturbations.

## Takeaways

This was definitely a very fun and interesting challenge! I got to learn more about the cat-and-mouse game of watermark insertion and removal, and play around with diffusion models. While the competition itself encourages a joint score of image degradation and low detection rates, I can see a more practical adversary caring way more about the former - after all, one can always try multiple times to bypass filtering (if, say, uploading to OSM platforms) while minimizing image degradation.

My solution is available here: [https://github.com/iamgroot42/adversarial-rinsing](https://github.com/iamgroot42/adversarial-rinsing)
