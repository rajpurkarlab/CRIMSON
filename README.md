# CRIMSON: A Clinically-Grounded LLM-Based Metric for Generative Radiology Report Evaluation

[Paper](https://arxiv.org/abs/2603.06183) | [Model (HuggingFace)](https://huggingface.co/CRIMSONScore/medgemma-4b-it-crimson)

CRIMSON is a clinically grounded evaluation framework for chest X-ray report generation that assesses reports based on diagnostic correctness, contextual relevance, and patient safety. Unlike prior metrics, CRIMSON incorporates full clinical context, including patient age, indication, and guideline-based decision rules. CRIMSON evaluates only abnormal findings, excluding normal findings from scoring. The framework categorizes errors into a comprehensive taxonomy covering false findings, missing findings, and eight attribute-level errors (e.g., location, severity, measurement, and diagnostic overinterpretation). Each finding is assigned a clinical significance level (urgent, actionable non-urgent, non-actionable, or expected/benign), based on a guideline developed in collaboration with attending cardiothoracic radiologists, enabling severity-aware weighting that prioritizes clinically consequential mistakes over benign discrepancies. Findings are weighted as 1.0 (urgent), 0.5 (actionable non-urgent), 0.25 (non-actionable), or 0.0 (expected/benign), and attribute errors are weighted as 0.5 (significant) or 0.0 (negligible). After weighting, the framework produces a score in the range of (-1, 1], where 1 represents a perfect report, 0 indicates the report is no more informative than a normal template, and negative scores indicate more weighted errors than correct findings.

## Installation

```bash
git clone https://github.com/rajpurkarlab/CRIMSON.git
cd CRIMSON
conda create -n crimson python=3.11
conda activate crimson
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

To also evaluate with RadEval metrics (optional, used by RadPref and RadJudge evaluation scripts), follow the installation instructions in the [RadEval repository](https://github.com/jbdel/RadEval).

## Usage

By default, CRIMSON uses the fine-tuned [CRIMSON-MedGemma model](https://huggingface.co/CRIMSONScore/medgemma-4b-it-crimson) via HuggingFace.

To use OpenAI GPT models instead, set your API key:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Scoring a report pair

```python
from CRIMSON.generate_score import CRIMSONScore

# Default: uses the HuggingFace MedGemmaCRIMSON model 
scorer = CRIMSONScore()
result = scorer.evaluate(
    reference_findings="Cardiomegaly. Small bilateral pleural effusions.",
    predicted_findings="Normal heart size. Small left pleural effusion.",
)

print(f"CRIMSON Score: {result['crimson_score']:.2f}")
print(f"False findings: {result['crimson_false_findings']}")
print(f"Missing findings: {result['crimson_missing_findings']}")
print(f"Attribute errors: {result['crimson_attribute_errors']}")
```

### Scoring with patient context

Providing patient context (age, indication) enables context-dependent clinical significance assignment — for example, missing aortic atherosclerosis in an 82-year-old with routine preoperative evaluation is expected/benign, but in a 25-year-old with chest pain it may be actionable.

```python
result = scorer.evaluate(
    reference_findings="Bibasilar atelectasis. Mild cardiomegaly. Aortic atherosclerosis with vascular calcification.",
    predicted_findings="Bibasilar atelectasis. Mild cardiomegaly.",
    patient_context={
        "Age": "82",
        "Indication": "Routine preoperative evaluation",
    },
)
```

### Using an OpenAI model (requires API key)

```python
scorer = CRIMSONScore(api="openai", model_name="gpt-5.2")
result = scorer.evaluate(reference_findings="...", predicted_findings="...")
```

## Components

| Directory | Description |
|---|---|
| `CRIMSON/` | Core scoring module: prompt construction (`prompt_parts.py`) and score generation (`generate_score.py`) |
| `RadPref/` | RadPref preference benchmark with ground truth + candidate reports, radiologist annotations, and evaluation scripts |
| `RadJudge/` | Curated ranking test suite with 30 clinically challenging test cases |
| `evaluate_reports.py` | End-to-end evaluation combining RadEval metrics and CRIMSON |

## RadPref

RadPref is a radiologist preference benchmark where 3 board-certified cardiothoracic radiologists compare pairs of candidate chest X-ray reports against ground-truth findings. Each rater provides:
- Per-finding error labels (false findings, missing findings, attribute errors)
- Clinical significance ratings per finding
- Overall quality scores (1–5)

The dataset is located in `RadPref/preference_data.json` with per-rater annotations in `RadPref/radiologist_annotations/`.

### Evaluating on RadPref

The evaluation script scores all candidate pairs using CRIMSON (via parallel API calls) and optionally [RadEval](https://github.com/jbdel/RadEval) metrics:

```bash
# Score with both CRIMSON and RadEval (default)
python RadPref/evaluate_radpref.py --input RadPref/preference_data.json --output scores.json

# Score with CRIMSON only
python RadPref/evaluate_radpref.py --input RadPref/preference_data.json --output scores.json --skip-radeval

# Score with RadEval only
python RadPref/evaluate_radpref.py --input RadPref/preference_data.json --output scores.json --skip-crimson
```

## RadJudge

RadJudge is a curated test suite of 30 ranking cases organized by error category (false findings, missing findings, attribute errors, clinical significance). Each case provides a ground truth and two or more candidates with a known expected ranking, verifying that metrics behave according to clinical intuition.

### Running RadJudge tests

```bash
# Run all tests with CRIMSON
python RadJudge/test_suite.py

# Run specific test cases
python RadJudge/test_suite.py --tests 1a 2b 3c

# Also compare against RadEval metrics
python RadJudge/test_suite.py --radeval
```

## Reference

```bibtex
@article{baharoon2025crimson,
  title={CRIMSON: A Clinically-Grounded LLM-Based Metric for Generative Radiology Report Evaluation},
  author={Baharoon, Mohammed and Heintz, Thibault and Raissi, Siavash and Alabbad, Mahmoud and Alhammad, Mona and AlOmaish, Hassan and Kim, Sung Eun and Banerjee, Oishi and Rajpurkar, Pranav},
  journal={arXiv preprint arXiv:2603.06183},
  year={2026}
}
```

If you use RadEval metrics in your evaluation, please also cite:

```bibtex
@inproceedings{xu2025radeval,
    title={RadEval: A framework for radiology text evaluation},
    author={Xu, Justin and Zhang, Xi and Abderezaei, Javid and Bauml, Julie and Boodoo, Roger and Haghighi, Fatemeh and Ganjizadeh, Ali and Brattain, Eric and Van Veen, Dave and Meng, Zaiqiao and others},
    booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
    pages={546--557},
    year={2025}
}
```
