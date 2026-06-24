# Supplementary LR re-ranking benchmarks

## Methods

To evaluate LR re-ranking under controlled conditions, we used two
complementary benchmarks. First, artificial focal interface, global
high-coverage, and broadly distributed immune-like LR programs were inserted
into the CID44971 spatial backbone while preserving its coordinates,
deconvolved cell-type composition, and reconstructed expression background.
Five expression scales and five random seeds were evaluated. Second, a fully
synthetic benchmark containing local, expression-matched diffuse, spatially
separated, global high-coverage, and random-background LR programs was
evaluated across 20 prespecified random seeds. Attention-based rankings were
compared with rankings based on candidate-edge abundance.

## Results

In the CID44971 semi-synthetic benchmark, focal interface programs shifted
from a median abundance rank of 13 to a median attention rank of 8, and 88% of
the 25 observations were promoted (one-sided Wilcoxon test,
P = 4.78 x 10^-5; Supplementary Fig. S4). In contrast, the global
CD99-like programs showed a median rank change of -5.

Across the 20 fully synthetic datasets, attention ranking achieved higher
candidate-set AUPRC than abundance ranking in every random seed (mean 0.303
versus 0.241; paired one-sided Wilcoxon test, P = 9.54 x 10^-7).
Global high-coverage controls shifted from a median abundance rank of 3.5 to a
median attention rank of 23 (Supplementary Fig. S5). However, local programs
did not consistently outrank expression-matched diffuse controls
(17/160 comparisons; median attention difference = -0.103). Thus, the
synthetic benchmark supports suppression of abundance-dominated global
signals, but also indicates that spatial focality alone is insufficient for
prioritization.

## Supplementary Figures

![Supplementary Figure S4](../../results/archetype_lr_cid44971_multiseed/archetype_benchmark_figure.png)

**Supplementary Figure S4. Real-data-backed semi-synthetic LR re-ranking
benchmark using the CID44971 spatial backbone.** Artificial LR programs
represented a focal CAF-tumor interface pattern, a globally distributed
CD99-like pattern, and a broadly distributed immune-like pattern. Focal
programs were generally promoted by attention-based re-ranking, whereas the
global high-coverage control was deprioritized.

![Supplementary Figure S5](../../results/synthetic_lr_v2_context_20seeds_figure/synthetic_lr_v2_20seeds.png)

**Supplementary Figure S5. Fully synthetic LR re-ranking benchmark across 20
prespecified random seeds.** (a) Candidate-set AUPRC for attention and
candidate-edge-abundance rankings. (b) Paired attention-score differences
between local programs and expression-matched diffuse controls. (c) Abundance
and attention ranks of global high-coverage controls. Attention ranking
consistently outperformed abundance ranking and suppressed global
high-coverage signals, whereas local-versus-diffuse discrimination was not
reliably recovered.
