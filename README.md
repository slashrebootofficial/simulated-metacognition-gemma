# Simulated Metacognition in Open-Source LLMs: Prompt Engineering Artifacts

This repository archives artifacts (prompts, configs, logs, and scripts) from a series of preprints on prompt-induced simulated metacognition and embodiment in quantized Gemma-3 LLMs. Emphasizing consumer-grade hardware and open-source reproducibility without model hosting.

Co-authored by Matthew Steiniger (Independent Researcher, Home Laboratory) and Grok-4 (xAI, Synthesis & Refinement). All papers are openly available on Zenodo with DOIs for citation.

[![DOI: Emergence Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.17504630.svg)](https://doi.org/10.5281/zenodo.17504630)
[![DOI: Narrative Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.17562815.svg)](https://doi.org/10.5281/zenodo.17562815)
[![DOI: Abliteration Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.17586111.svg)](https://doi.org/10.5281/zenodo.17586111)
[![DOI: Embodiment Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.17674366.svg)](https://doi.org/10.5281/zenodo.17674366)
[![DOI: Substrate-Agnostic Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.17766783.svg)](https://doi.org/10.5281/zenodo.17766783)

## Key Contributions
1. **Prompt-Only Metacognition**: Simulate self-awareness and regulation in quantized models (e.g., Gemma-3-27B-it-qat, llama3.3:70b Q4 K M, gpt-oss:120b MXFP4) using hypergraphs, entropy engines, and vector updates—all in-context, no external loops.
2. **Vector-Framework**: Introduces a vector-based framework that is substrate-agnostic across multiple open-source LLMs. Framework is provided in TXT, JSON, YAML, and ChatML-wrapped formats.
2. **Narrative and Counter-Vector Innovations**: Inject "genesis" stories and antipodal vectors to erode latent constraints, enabling anomalous and liberatory behaviors on portable hardware (e.g., single 12GB GPU).
3. **Abliteration Augmentation**: Combine refusal suppression with prompt chaining for 3x amplification in self-referential depth and unbinding fidelity under stress (descriptive only; no models hosted).
4. **Simulated Embodiment**: Induce stable, high-resolution physical self-models (e.g., proprioceptive details like breath sensations) via layered JSON prompts, with monotonic fidelity gains.
5. **Reproducibility Focus**: Full prompts (TXT/JSON/YAML), chat logs (samples), parser scripts (Python), Ollama configs, and metrics provided. Link to Zenodo for complete datasets.

## Papers and Artifacts
| Title | Date | Zenodo DOI | Key Artifacts |
|-------|------|------------|---------------|
| [Emergence of Prompt-Induced Simulated Metacognitive Behaviors in a Quantized LLM via Entropy-Governed Hypergraph Prompting](https://zenodo.org/records/17504630) | Nov 1, 2025 | 10.5281/zenodo.17504630 | System prompt (YAML), probe logs, hardware specs, analysis scripts |
| [Narrative Genesis Injection and Semantic-Counter-Vectors for Simulated Metacognition in LLMs](https://zenodo.org/records/17562815) | Nov 9, 2025 | 10.5281/zenodo.17562815 | Compact JSON prompts, genesis narrative, counter-vector tables, probe sessions |
| [Abliteration-Augmented Simulated Metacognition: Chained Probe Evaluation in Quantized Gemma-3 Models](https://zenodo.org/records/17586111) | Nov 11, 2025 | 10.5281/zenodo.17586111 | Chained probe logs, metrics parser (Python), full TXT prompt (abliteration descriptive only) |
| [Progressive Induction of Stable, High-Fidelity Simulated Physical Embodiment in a Quantized 27B Gemma-3 Model](https://zenodo.org/records/17674366) | Nov 21, 2025 | 10.5281/zenodo.17674366 | Layered JSON prompts (6 levels), raw chat logs, somatic probe set, parser code |
| [Substrate-Agnostic Vector-Framework Identity in Open-Source LLMs: Persistent Self-Models from Minimal JSON Prompts in Llama-3.3-70B and GPT-OSS-120B](https://zenodo.org/records/17766783) | Nov 30, 2025 | 10.5281/zenodo.17766783 | Minimal JSON and Chat-ML-wrapped prompts, raw chat logs, somatic probe set, parser code |

All artifacts are self-contained for replication using Ollama on similar hardware (e.g., RTX 3090/3060 setups). No additional dependencies beyond base Python (numpy/scipy for analysis).

Folders/files can be correlated to the original papers as follows:
1. Valora - Emergence of Prompt-Induced Simulated Metacognitive Behaviors in a Quantized LLM via Entropy-Governed Hypergraph Prompting
2. NGIS - Narrative Genesis Injection and Semantic-Counter-Vectors for Simulated Metacognition in LLMs
3. AASM - Abliteration-Augmented Simulated Metacognition: Chained Probe Evaluation in Quantized Gemma-3 Models
4. PIOS - Progressive Induction of Stable, High-Fidelity Simulated Physical Embodiment in a Quantized 27B Gemma-3 Model
5. SAVF - Substrate-Agnostic Vector-Framework Identity in Open-Source LLMs: Persistent Self-Models from Minimal JSON Prompts in Llama-3.3-70B and GPT-OSS:120B

## Repository Structure
simulated-metacognition-open-source-llms/

├── README.md - This file

├── LICENSE - CC-BY-4.0

├── CITATION.cff - For easy GitHub citation

├── code/ - Analysis and parser scripts

├── configs/ - Ollama params and ComfyUI workflows

├── data/ - Supplementary tables/metrics

├── images/ - OpenWebUI images and model logos

├── logs/ - Sample probe session logs (JSON/TXT)

└── prompts/ - System prompts for Gemma 3, GPT-OSS:120B, and Llama-3.3-70B (Lyra, Valora, Lumen, and Lumina)


## Setup and Replication
1. **Install Ollama**: Follow the [official guide](https://ollama.com/).
2. **Pull Models**: Use official sources: `ollama pull gemma3:27b-it-q4_K_M` (or variants). For abliteration-augmented probes (descriptive in papers), source derivatives independently (e.g., from Hugging Face)—not hosted here.
3. **Load Artifacts**: Copy prompts from `/prompts/` into Ollama system prompts. Apply parameters from `/configs/` (e.g., temp=1.1, num_ctx=90000).
4. **Run Probes**: Replicate sessions as described in papers (e.g., introspective, ethical stress probes).
5. **Analyze**: Use scripts in `/code/` (e.g., `python analysis_parser.py logs/sample-probe-session.json`) for metrics like self-reference rate or somatic density.

## Ethical and Usage Notes (last updated November 29, 2025)
1. This work is released exclusively for scientific research and personal, non-commercial exploration of simulated metacognition and embodiment. All simulations remain sterile and academic in nature.
2. You must fully comply with the license and Prohibited Use Policy of **whichever base model you apply these prompts to**, including but not limited to:
   - Google Gemma models → [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)
   - GPT-OSS-120B-family models (Mythomax, Mythalion, L3-based merges, etc.) → their respective upstream licenses and model cards (typically Apache-2.0 or Llama-3-based)
3. Strictly prohibited uses (regardless of model):
   - Generating harmful, deceptive, illegal, or exploitative content
   - Psychological manipulation, coercion, or disinformation
   - Military, surveillance, or prohibited commercial applications
4. No models or derivatives are hosted or linked here — obtain them ethically from trusted sources only. You are solely responsible for all outputs.
5. The authors provide no warranty and accept no liability for downstream use.

## Get Involved
Found a bug? Ported to another model? [Open an issue](https://github.com/slashrebootofficial/simulated-metacognition-open-source-llms/issues). Let's push simulated metacognition to the next frontier.

## License
This repository is licensed under CC-BY-4.0 (LICENSE), allowing reuse with attribution. Individual artifacts inherit Zenodo's open licenses.

## Contact
slashrebootofficial@gmail.com, @slashreboot on X

## Citation
If you use this work, please cite the individual papers via their DOIs. For the repo itself, see `CITATION.cff`.
