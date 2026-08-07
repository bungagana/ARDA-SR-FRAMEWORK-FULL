# Cross-Backbone ARDA-SR Experiments

This folder runs ARDA-SR with hosted Hugging Face Inference backbones. It is
intended for robustness checks beyond the Gemini-only main experiment without
running heavy models locally.

## Models

Configured hosted candidates:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`

You can swap in any Hugging Face text-generation model available through
Inference Providers. Free availability depends on Hugging Face account limits
and provider availability.

## Setup

Set a Hugging Face token. A free HF account token is usually enough for
rate-limited inference:

```powershell
$env:HF_TOKEN="hf_..."
```

## Smoke Test

Run two queries per category:

```powershell
python cross\run_cross_backbone.py --smoke
```

Run explicit models:

```powershell
python cross\run_cross_backbone.py --smoke --models Qwen/Qwen2.5-1.5B-Instruct,microsoft/Phi-3-mini-4k-instruct
```

## Full Run

This is much slower because ARDA-SR uses routing, dual-draft arbitration, PMR, and
optional judging:

```powershell
python cross\run_cross_backbone.py --judge
```

Without `--judge`, answer-quality metrics (`Rel`, `Faith`, `Cov`) are not computed;
automatic metrics such as `Hit@5`, `CtxRel`, `ToolAcc`, `SRComp`, `FRR`, `FAR`, and
latency are still produced.

## Outputs

Files are written to `cross/results/`:

- `{model}_arda_pmr_results.json`
- `cross_backbone_metrics.json`
- `cross_backbone_summary.csv`
- `table_cross_backbone.tex` after running `python cross\analyze_cross_results.py`

## Paper Wording

Use conservative wording:

> We conduct an additional cross-backbone robustness check using hosted
> Hugging Face inference backbones. These results are treated as supplementary because the main
> experiments use Gemini 2.5 Flash-Lite under a controlled single-backbone setting.

## Troubleshooting

If Python exits with a Windows access-violation while importing the ARDA-SR
pipeline, the issue is usually the local scientific stack import path
(`sentence_transformers` -> `sklearn/pandas/pyarrow`), not the Hugging Face
client.

The Hugging Face adapter can be tested independently:

```powershell
python -c "from cross.hf_client import HFInferenceClient; print(HFInferenceClient('Qwen/Qwen2.5-1.5B-Instruct').generate('Reply OK', max_tokens=8))"
```

Use a clean virtual environment with compatible `pandas`, `pyarrow`,
`scikit-learn`, and `sentence-transformers` versions before running the full
pipeline.

The previous Ollama path is still available as a fallback:

```powershell
python cross\run_cross_backbone.py --provider ollama --smoke --models gemma3:1b
```
