"""
Prompt template for generating case embeddings for similarity clustering.
Uses GPT-5.4 to produce a structured clinical summary suitable for embedding.
"""

CASE_SUMMARY_SYSTEM = """You are a clinical case classifier. Given a clinical case
presentation, produce a structured summary that captures the key clinical dimensions
for similarity comparison.

Return ONLY valid JSON:

```json
{
  "organ_systems": ["renal", "hematologic"],
  "primary_presenting_features": ["nephrotic syndrome", "weight loss", "edema"],
  "key_lab_abnormalities": ["proteinuria", "hypoalbuminemia"],
  "demographic_profile": "elderly male with diabetes",
  "diagnostic_category": "glomerular disease",
  "clinical_complexity": "high",
  "reasoning_challenge_type": "anchoring_risk"
}
```

For `reasoning_challenge_type`, classify as one of:
- **anchoring_risk**: an obvious diagnosis exists but key features suggest something else
- **broad_differential**: many plausible diagnoses, requires systematic narrowing
- **rare_diagnosis**: the correct answer is uncommon and easily missed
- **atypical_presentation**: a common disease presenting unusually
- **multisystem**: features span multiple organ systems, requiring integration
"""

CASE_SUMMARY_USER = """## Case Presentation

{case_presentation}

## Correct Diagnosis

{correct_diagnosis}

---

Produce the structured clinical summary as JSON. Return ONLY the JSON object.
"""
