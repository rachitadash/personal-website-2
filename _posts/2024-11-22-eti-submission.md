---
layout: distill
title: My submission to the ETI Challenge
description: Description of my entry to the ETI (Erasing the Invisible) challenge (co-located with NeurIPS) for watermark-removal.
# tags: watermark removal, adverarial examples, diffusion models
# giscus_comments: true
date: 2024-11-12
citation: true
featured: true

authors:
  - name: Anshuman Suri
    url: "https://anshumansuri.com/"
    affiliations:
      name: Northeastern University

bibliography: 2024-11-22-eti-submission.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
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

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
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


---

## Interactive Plots

You can add interative plots using plotly + iframes :framed_picture:

<div class="l-page">
  <iframe src="{{ '/assets/plotly/demo.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>

The plot must be generated separately and saved into an HTML file.
To generate the plot that you see above, you can use the following code snippet:

{% highlight python %}
import pandas as pd
import plotly.express as px
df = pd.read_csv(
'https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv'
)
fig = px.density_mapbox(
df,
lat='Latitude',
lon='Longitude',
z='Magnitude',
radius=10,
center=dict(lat=0, lon=180),
zoom=0,
mapbox_style="stamen-terrain",
)
fig.show()
fig.write_html('assets/plotly/demo.html')
{% endhighlight %}

---

## Details boxes

Details boxes are collapsible boxes which hide additional information from the user. They can be added with the `details` liquid tag:

{% details Click here to know more %}
Additional details, where math $$ 2x - 1 $$ and `code` is rendered correctly.
{% enddetails %}

---

## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body` sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

---

## Other Typography?

Emphasis, aka italics, with _asterisks_ (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or **underscores**.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
   ⋅⋅\* Unordered sub-list.
3. Actual numbers don't matter, just that it's a number
   ⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behaviour, where trailing spaces are not required.)

- Unordered list can use asterisks

* Or minuses

- Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links.
http://www.example.com or <http://www.example.com> and sometimes
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style:
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style:
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```

```python
s = "Python syntax highlighting"
print s
```

```
No language indicated, so no syntax highlighting.
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        |      Are      |  Cool |
| ------------- | :-----------: | ----: |
| col 3 is      | right-aligned | $1600 |
| col 2 is      |   centered    |   $12 |
| zebra stripes |   are neat    |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the
raw Markdown line up prettily. You can also use inline Markdown.

| Markdown | Less      | Pretty     |
| -------- | --------- | ---------- |
| _Still_  | `renders` | **nicely** |
| 1        | 2         | 3          |

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can _put_ **Markdown** into a blockquote.

Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a _separate paragraph_.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the _same paragraph_.
