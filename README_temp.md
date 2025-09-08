Where your code matches CAPGen

Color-probability parameterization. You learn a per-pixel categorical distribution over K base colors (logits → softmax(·/τ)), then mix a fixed palette to produce RGB: exactly the paper’s “color probability matrix” idea. 
arXiv

Palette from environment colors via k-means. You extract base colors with K-means (paper’s default, too). 
arXiv

EOT-style robustness. You paste the patch with transformations and resize before the detector; the paper also frames training with EOT (see Appendix D). 
arXiv

Key differences vs. the paper

How you obtain the palette (data scope).

Yours: K-means on one background image (capgen_image_path).

Paper: K-means over a set of images from the target environment to better capture typical base colors. Consider letting capgen_image_path accept a folder/list and run K-means on all pixels concatenated. 
arXiv

Number of base colors K (defaults).

Yours: default K=6.

Paper: default K=3 (with an ablation showing more colors can help). You can stick with 6, but if you want to match their setup precisely set capgen_num_colors=3. 
arXiv

Sharpness/regularization toward one-color-per-pixel.

Paper: explicitly “regularizes the color probability matrix so each pixel belongs to just one base color” and fixes τ≈0.1 to keep assignments sharp. That’s effectively low entropy per pixel. 
arXiv

Yours: you added an entropy term with a negative sign

ent = -(r * (r + r_eps).log()).sum(-1).mean()  # entropy H(r)
ent_loss = -entropy_weight * ent               # encourages HIGH entropy


Minimizing loss + ent_loss with entropy_weight>0 increases entropy (more color blending), which is the opposite of their “near one-hot” regularization. If you want to match the paper’s intent, flip the sign:

ent_loss = +entropy_weight * ent  # penalize entropy → crisper, near one-hot


and/or keep tau small (≈0.1) without annealing. 
arXiv

Fast environment adaptation (pattern–color decoupling).

Paper: after you’ve learned the pattern (the probability matrix / logits), you can rapidly adapt to a new scene by keeping logits fixed and swapping the palette to base colors from the new environment—no re-optimization needed. 
arXiv

Yours: palette is fixed at init; there’s no “quick recolor” path. You can support it by exposing a method that replaces self.capgen_mod.palette with a new K-means palette and immediately renders the adapted patch using the existing logits.

Temperature schedule.

Paper: reports τ≈0.1 to keep pixels assigned to single base colors. 
arXiv

Yours: allows τ annealing (capgen_tau_start, capgen_tau_end). If you want closer fidelity, keep both at 0.1.

Loss composition.

Paper: discusses the trade-off between deception and stealth, formalizing EOT training and color-probability regularization. 
arXiv

Yours: adds NPS and TV terms from the Du et al. trainer—useful for printability/smoothness, but not called out in CAPGen. This is fine (and often helpful) but it’s an extension.
