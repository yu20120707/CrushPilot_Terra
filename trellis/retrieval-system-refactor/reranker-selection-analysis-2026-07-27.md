# Reranker Selection Analysis

## Scope

This is an offline model-selection diagnostic. Golden labels are used only by the evaluator;
they are not imported by the production planner, retrieval service, or ranking code.

## Evidence

The latest complete production-shaped Live Gate (before the Top15 backfill fix) reported:

| Strategy / plan source | Recall@20 | MRR | nDCG@10 | Avg selected |
| --- | ---: | ---: | ---: | ---: |
| Live planner + current CrossEncoder | 0.63095 | 0.24539 | 0.19043 | 3.92 |
| Gold plan + RRF-order control | 0.84226 | 0.29152 | 0.22962 | 5.575 |

The rows are not an A/B model comparison because their retrieval plans differ. The second row
is a retrieval ceiling diagnostic with Golden `required_topics` and `excluded_topics`; it shows
that:

1. preserving RRF order is directionally better than the current complete Live Gate;
2. even a correct topic plan does not reach the `Recall@20 >= 0.90` or `nDCG@10 >= 0.75` release
   thresholds;
3. reranker calibration alone cannot close the remaining upstream recall gap.

Existing complete traces also show gold + acceptable case hits falling from `95/112` at fused
Top20 to `62/112` after the old reranked Top6 and `47/112` after selection. The structural
Top6-before-Diversity defect is fixed separately.

## CrossEncoder replay limitation

An ad-hoc process that loaded both the embedding model and `BAAI/bge-reranker-base` for a
120-case gold-plan replay exceeded 10 minutes on CPU without producing a completed comparison.
The timed-out process was terminated explicitly. This result must not be represented as a
CrossEncoder metric.

## Next controlled experiment

After model service credit is restored:

1. run the formal Scene Plan Gate;
2. run the formal Live Gate with the Top15 backfill fix;
3. freeze fused Top15 and CrossEncoder ranks;
4. compare one global reciprocal-rank fusion weight across fixed folds;
5. reject the fusion unless validation recall, MRR/nDCG, forbidden rejection, boundary recall,
   and no-evidence metrics all satisfy the design gates.

No case, category, Golden chunk ID, or label may enter runtime ranking logic.
