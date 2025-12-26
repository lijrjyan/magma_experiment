# Dual Defense (DDFed) vs. MAGMA (Work-in-Progress)

Stage 1 establishes the engineering scaffolding for a reproducible MAGMA
benchmark. The detailed comparison between Dual Defense and MAGMA will be
expanded in later stages of the execution plan. For now we capture the
key axes that guide the refactor:

1. **Observation signal**: DDFed relies on global-to-client cosine
   similarities, whereas MAGMA will operate on pairwise manifold
   distances derived from last-layer differences or richer embeddings.
2. **Selection mechanism**: DDFed thresholds scalar scores via
   majority-vote feedback. MAGMA will reconstruct low-curvature
   components from hierarchical clustering (Ward + jump ratio) to decide
   which clients remain.
3. **False-positive philosophy**: MAGMA defaults to conservative
   "accept-all" when no jump gap exists, aligning with the
   "no-evidence-no-rejection" requirement in the execution plan.
4. **Interaction/computation**: MAGMA emphasizes structural recovery
   (higher compute, fewer interactive rounds) compared to DDFed's
   two-phase feedback loop.

Subsequent stages will add concrete experiment data, figures, and tables
that quantify these differences.
