# SGOOTR parameter profiles

The SGOOTR profile workflow reuses the shared profile grid, likelihood-threshold
interval calculation, and TLS figure renderer. It profiles one parameter at a
time around each fit and uses the stem-free crown likelihood with root state 0
known, conditioning on both daughter lineages surviving.

Run either profile target, or use the default workflow to generate both:

```bash
snakemake --cores 1 src/results/sgootr/figures/SGOOTR_composite_sensitivity_compact.pdf
snakemake --cores 1
```

The workflow writes five matching figure types in PDF and SVG under
`src/results/sgootr/figures/` and profile tables under
`src/results/sgootr/data/`. The middle panel of the composite figure shows the
individual transition probabilities `NC->PT` and `PT->LN`.
