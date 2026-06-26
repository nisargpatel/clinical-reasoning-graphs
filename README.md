# Clinical Reasoning Graphs

**Structured Evaluation of LLM Diagnostic Reasoning Reveals Competence Without Consistency**
(SD4H @ ICML 2026)

Code and released artifacts for extracting **clinical reasoning graphs** from free-text LLM
diagnostic traces and testing whether those traces show stable, schema-like reasoning structure
for clinically similar cases. Across 750 traces (50 NEJM CPC cases × 5 frontier models × 3 prompt
conditions), within-cluster and between-cluster graph similarity are statistically
indistinguishable: models reach accurate diagnoses without reasoning that is measurably more
similar for similar cases.

## What is a clinical reasoning graph?

A directed graph extracted from an LLM diagnostic trace using a domain-grounded ontology:

- **5 node types** — clinical feature, diagnosis (with confidence), semantic qualifier,
  discriminating feature, evidence reference.
- **7 edge types** — `supports`, `argues_against`, `discriminates_between`,
  `triggered_reflection`, and the post-reflection updates `promoted` / `demoted` / `unchanged`.
- **Phase tags** mark when each edge was established (initial assessment → reflection → revised),
  capturing the temporal structure of the reasoning.

## What's in this repository

| Path | Contents |
|---|---|
| `data/extracted/all_graphs.json` | The **750 extracted reasoning graphs** (verbatim NEJM case text removed). |
| `data/scored_results_public.csv` | Per-trace scoring (accuracy + chairman top-1/3/5 + metadata; the case prompt and full model responses are removed). |
| `data/clusters/` | Case clusters (`clusters.json`), case embeddings, and category metadata. |
| `data/gold_dx_embeddings.json` | Gold-diagnosis embeddings (local PubMedBERT, 768-dim) for the content-channel / Mantel regressions. |
| `data/analysis/*.npy` | Precomputed pairwise similarity arrays (within / between / test-retest / inter-extractor). |
| `results/*.json`, `results/discriminating_ratios.csv` | Shipped summary analysis artifacts. |
| `src/`, `analysis/` | Extraction pipeline, similarity metrics, and analysis code. |
| `prompts/` | Extraction prompt and the generation/scoring prompts (Appendix G). |
| `experiments/validation/` | 30 physician-reviewed validation traces. |
| `experiments/{gold_ablation,retest,extractor_comparison}/` | Reliability re-extractions (gold-ablation, test-retest, inter-extractor). |

**Not redistributed:** the underlying NEJM CPC **case presentations** (published in the *New
England Journal of Medicine*). Raw model traces and `data/raw/` are **not included**; they are
needed only to *re-run the upstream extraction step* (see below), not to reproduce the analyses.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the paper's numbers

All commands run from the repository root and operate on the released `all_graphs.json` — no API
keys or external data required. Each command below was verified to run end-to-end on a fresh clone.

```bash
# Primary consistency result — per-cell within/between similarity, pooled 0.475 / 0.472 (Table 1, App. D)
python -m analysis.run_analysis --step reproduce  --graphs data/extracted/all_graphs.json --clusters data/clusters/clusters.json

# Per-cell decomposition + content-channel sign tests
python -m analysis.run_analysis --step decompose  --graphs data/extracted/all_graphs.json --clusters data/clusters/clusters.json

# Content-channel Mantel regression — mean r = 0.118
python -m analysis.run_analysis --step mantel     --graphs data/extracted/all_graphs.json --clusters data/clusters/clusters.json

# Component ablation (Table 5), noise floor (0.466), per-cell permutation tests, difficulty tiers
python -m src.extended_analysis --only ablation     --graphs data/extracted/all_graphs.json --clusters data/clusters/clusters.json
python -m src.extended_analysis --only noise        --graphs data/extracted/all_graphs.json --clusters data/clusters/clusters.json
python -m src.extended_analysis --only permutation  --graphs data/extracted/all_graphs.json --clusters data/clusters/clusters.json
python -m src.extended_analysis --only difficulty   --graphs data/extracted/all_graphs.json --clusters data/clusters/clusters.json

# Accuracy-vs-similarity orthogonality (§3.2; 0.488 both-correct vs 0.484 both-incorrect) — uses the released CSV
python -m src.extended_analysis --only accuracy     --graphs data/extracted/all_graphs.json --clusters data/clusters/clusters.json --scored data/scored_results_public.csv

# Calibration table (Table 4): composite + content-only bounds, test-retest, inter-extractor, gold-ablation
python analysis/calibrate_content.py
```

Reproduction runs write only to `results/extended/` (gitignored); the shipped summary artifacts at
`results/` root are not overwritten.

## Numbers that require inputs not in this tree (honest provenance)

- **Soft-Jaccard / semantic-matching row (Table 5, ≈ 0.003 / 0.001):** requires
  `data/extracted/label_embeddings.json` (OpenAI `text-embedding-3-large`, ≈ 2 GB), which is **not
  redistributed**. Regenerate the cache, then:
  ```bash
  python -m src.extended_analysis --only semantic --graphs data/extracted/all_graphs.json \
      --clusters data/clusters/clusters.json --embeddings data/extracted/label_embeddings.json
  ```
  Without `--embeddings`, this analysis is skipped.
- **Extraction-fidelity / physician-review figures (§2.3: 98.4% edge precision, 94.8% recall):** a
  manual physician review of the 30 sampled traces in `experiments/validation/`, **not produced by
  a rerunnable script**.
- **Structured-reflection within-trace stats (§3.4: 66.3 vs 57.0 motifs/graph; discriminating
  motifs +33%, p < 10⁻⁶):** computed by **`src.analyze`** (analyses 5–6, via `analysis/motifs.py`:
  `motif_distribution_by_condition`, `discriminating_motif_ratio`); the resulting per-condition
  values are shipped in `results/motif_distributions.json` and `results/discriminating_ratios.csv`.
- **"60–70% diagnostic accuracy" (Introduction):** an external benchmark figure cited from prior
  work, **not computed in this repository**.
- **Generation and scoring parameters (Appendix G):** in
  `prompts/generation_and_scoring_prompts.md`.

## Re-running the upstream extraction (optional — requires raw traces)

The extracted graphs are provided, so this is not needed to reproduce the analyses. To regenerate
them you must supply the raw model traces and the NEJM case texts under `data/raw/` (neither is
redistributed), set an OpenAI key in `.env` (the extractor is GPT-5.4), then:

```bash
python -m src.extract --input data/raw/results.jsonl --cases data/raw/cases.json --output data/extracted/
```

## Models and conditions

| Model | Provider | Role |
|---|---|---|
| GPT-5.4 | OpenAI | Subject + graph extractor |
| GPT-5.2 | OpenAI | Subject |
| Claude Opus 4.5 | Anthropic | Subject |
| Claude Sonnet 4.5 | Anthropic | Subject |
| Gemini 3 Pro Preview | Google | Subject |

| Condition | Description |
|---|---|
| Baseline | Single-pass differential + probabilities |
| Adversarial | Argue against the leading diagnosis, then revise |
| Structured reflection | ART framework: problem representation → balanced reassessment → defend/update |

## Companion study

The 750 traces were generated under the preregistered protocol for a companion
diagnostic-accuracy / confidence-calibration study (preregistration: OSF, see the paper's
references).

## Citation

```bibtex
@inproceedings{patel2026crg,
  title     = {Clinical Reasoning Graphs: Structured Evaluation of LLM Diagnostic
               Reasoning Reveals Competence Without Consistency},
  author    = {Patel, Nisarg},
  booktitle = {Workshop on Structured Data for Health (SD4H) at ICML},
  year      = {2026}
}
```

## License

MIT
