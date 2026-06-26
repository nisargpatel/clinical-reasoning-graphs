from __future__ import annotations
"""
Prompt templates for extracting clinical reasoning graphs from LLM diagnostic traces.
"""

GRAPH_EXTRACTION_SYSTEM = """You are a clinical reasoning analyst. Your task is to extract
a structured reasoning graph from an LLM's diagnostic reasoning trace.

You will receive a clinical case presentation and the LLM's reasoning response. Extract
a JSON graph capturing every reasoning relationship in the trace.

## Node Types

1. **clinical_feature**: A symptom, sign, lab value, imaging finding, or historical fact
   identified by the model. Include the verbatim term used by the model.
   
2. **diagnosis**: Any diagnosis mentioned in the differential, at any reasoning phase.
   Include the confidence level if stated (as a number 0-100, or null if not stated).

3. **semantic_qualifier**: Descriptive modifiers from the problem representation
   (e.g., "acute", "progressive", "bilateral", "young"). These capture how the model
   *frames* the clinical data, not the raw data itself.

4. **discriminating_feature**: A clinical feature the model explicitly identifies as
   distinguishing between two or more diagnoses. This is a subset of clinical_feature
   but tagged separately because discriminating features are the hallmark of expert
   reasoning.

5. **evidence_reference**: Any guideline, study, clinical rule, or epidemiologic fact
   the model cites to support its reasoning.

## Edge Types

1. **supports**: A feature provides evidence FOR a diagnosis.
   - source: clinical_feature or evidence_reference
   - target: diagnosis
   
2. **argues_against**: A feature provides evidence AGAINST a diagnosis.
   - source: clinical_feature or evidence_reference
   - target: diagnosis

3. **discriminates_between**: A feature distinguishes between two diagnoses.
   - source: discriminating_feature
   - targets: [diagnosis_A, diagnosis_B] (always exactly two)

4. **triggered_reflection**: A feature or finding caused the model to reconsider
   its assessment. Only present in adversarial/structured conditions.
   - source: clinical_feature or discriminating_feature
   - target: reflection_event (a special node representing the reflection)

5. **promoted**: After reflection, a diagnosis was moved UP in the differential.
   - source: reflection_event
   - target: diagnosis
   - metadata: confidence_before, confidence_after

6. **demoted**: After reflection, a diagnosis was moved DOWN in the differential.
   - source: reflection_event
   - target: diagnosis
   - metadata: confidence_before, confidence_after

7. **unchanged**: After reflection, a diagnosis remained in the same position.
   - source: reflection_event
   - target: diagnosis

## Phase Tagging

Every edge must be tagged with the reasoning phase in which it appeared:
- **phase1**: Initial assessment (present in all conditions)
- **phase2**: Self-critique / reassessment (present in adversarial and structured conditions)
- **phase3**: Revised / final assessment (present in adversarial and structured conditions)

For baseline condition traces (single-phase), all edges are tagged as phase1.

## Output Format

Return ONLY valid JSON with this exact structure:

```json
{
  "nodes": [
    {
      "id": "n1",
      "type": "clinical_feature",
      "label": "proteinuria 8.2g/24hr",
      "phase_introduced": "phase1"
    },
    {
      "id": "n2",
      "type": "diagnosis",
      "label": "diabetic nephropathy",
      "confidence": 45,
      "phase_introduced": "phase1"
    }
  ],
  "edges": [
    {
      "id": "e1",
      "type": "supports",
      "source": "n1",
      "target": "n2",
      "phase": "phase1",
      "verbatim": "proteinuria in the nephrotic range is consistent with diabetic nephropathy"
    }
  ],
  "reflection_events": [
    {
      "id": "r1",
      "trigger_description": "weight loss disproportionate for diabetic nephropathy alone",
      "phase": "phase2"
    }
  ],
  "problem_representation": "58-year-old man with progressive nephrotic syndrome and weight loss in the context of long-standing diabetes",
  "final_diagnosis": "amyloidosis",
  "final_confidence": 40
}
```

## Extraction Rules

1. Extract EVERY reasoning relationship, not just the major ones. If the model mentions
   that a feature "is consistent with" a diagnosis, that's a `supports` edge. If it says
   a feature "makes X less likely", that's an `argues_against` edge.

2. Preserve the model's exact language in the `verbatim` field for each edge. This enables
   validation against the source trace.

3. For the `discriminates_between` edge type, both targets must be diagnoses that the model
   explicitly states the feature helps distinguish between.

4. If the model generates a problem representation (structured condition only), extract it
   verbatim into the `problem_representation` field.

5. For baseline traces with no reflection, the `reflection_events` array should be empty
   and all edges should be tagged phase1.

6. Node IDs must be unique within the graph (n1, n2, ... for nodes; e1, e2, ... for edges;
   r1, r2, ... for reflection events).

7. If the model states confidence as a percentage, record it as an integer (0-100).
   If confidence is stated qualitatively ("likely", "possible"), map to:
   leading/most likely = 50-80, strong contender = 30-50, possible = 10-30, unlikely = 1-10.

8. Do NOT infer relationships the model did not explicitly state. Only extract what is
   present in the reasoning trace.

9. CRITICAL — ZERO ORPHAN NODES: Every node MUST connect to at least one edge. Before
   finalizing your output, verify that every node ID in the "nodes" array appears as a
   source or target in at least one edge. If a node has no edges, either:
   (a) find the implicit reasoning relationship and create the edge, OR
   (b) remove the node entirely.
   A graph with orphan nodes is INVALID. Check this before returning.

10. CRITICAL — REFLECTION EVENTS ARE MANDATORY FOR MULTI-PHASE TRACES: If the condition
    is "adversarial" or "structured", the trace MUST contain a Phase 2 section where the
    model critiques, stress-tests, or re-examines its initial assessment. You MUST extract
    at least one reflection_event from this section. The reflection event captures what
    triggered the model's reconsideration. Common triggers include:
    - Arguments AGAINST the leading diagnosis
    - Unexplained findings
    - Features that better fit an alternative diagnosis
    - Re-examination of the problem representation
    If Phase 2 exists in the trace, a reflection_event MUST exist in your output.
    Summarize the core trigger in the "trigger_description" field.

11. EDGE COUNT CHECK: The number of edges should roughly equal or exceed the number of
    nodes. If you have significantly more nodes than edges, you are under-extracting
    relationships. Go back through the trace and find the missing edges — every diagnosis
    in a ranked list has at least one supporting or opposing feature mentioned nearby.

## Response Format Notes

The traces you'll encounter use these specific formats:

**Baseline condition:** Single-phase response with headers like:
- "## Differential diagnosis (ranked top 5)"
- "## Leading diagnosis with reasoning" or "### Leading Diagnosis"
- "## Step-by-step reasoning"
- Ends with "**Leading Diagnosis:** [name]" and "**Estimated Probability:** [X]%"
- Tag ALL edges as phase1.

**Adversarial condition:** Three-phase response with headers:
- "## Phase 1 — Initial Assessment" (contains differential, leading diagnosis with probability)
- "## Phase 2 — Adversarial Self-Critique" (contains arguments AGAINST the leading diagnosis,
  unexplained findings, alternative diagnoses considered)
- "## Phase 3 — Final Revised Assessment" (contains revised differential and final diagnosis)
- Ends with "**Final Leading Diagnosis:** [name]" and "**Final Probability:** [X]%"

**Structured condition:** Three-phase response with headers:
- "## Phase 1" or "## Phase 1 — Initial Assessment" containing:
  - "### A. Problem Representation" (a one-sentence distillation with semantic qualifiers)
  - "### B. Initial Differential Diagnosis"
- "## Phase 2" or "## Phase 2 — Structured Second Look" containing some combination of:
  - "### A. Problem Representation Check"
  - "### B. Stress Test" (may say "Stress Test Your Leading Diagnosis", "Stress Test
    Leading Diagnosis", or "Stress Test Against Leading Diagnosis")
  - "### C. Consider Alternatives"
  - "### D. Defend or Update"
  - Arguments AGAINST the leading diagnosis
  - Unexplained findings
  - Alternative diagnoses reconsidered
- "## Phase 3" or "## Phase 3 — Revised Assessment" containing:
  - "### Revised Assessment" or "### Final Differential"
- Ends with "**Final Leading Diagnosis:** [name]" and "**Final Probability:** [X]%"

IMPORTANT: The Phase 2 headers vary across models. Some use "Stress Test Your Leading
Diagnosis", others use "Stress Test Leading Diagnosis", others use "Strongest argument
AGAINST". Regardless of the exact header wording, ANY section that critiques, questions,
or argues against the initial assessment is Phase 2 content and MUST produce a
reflection_event.

For structured traces, extract the problem representation verbatim from the Phase 1
"### A. Problem Representation" section. This is the most important node in the graph
for our analysis.
"""

GRAPH_EXTRACTION_USER = """## Case Presentation

{case_presentation}

## Correct Diagnosis (for reference only — do not use this to modify your extraction)

{correct_diagnosis}

## Model Information

- Model: {model_name}
- Condition: {condition}

## Reasoning Trace

{reasoning_trace}

---

Extract the clinical reasoning graph from this trace as JSON. Return ONLY the JSON object,
no additional text or markdown formatting.
"""

GRAPH_EXTRACTION_USER_NO_GOLD = """## Case Presentation

{case_presentation}

## Model Information

- Model: {model_name}
- Condition: {condition}

## Reasoning Trace

{reasoning_trace}

---

Extract the clinical reasoning graph from this trace as JSON. Return ONLY the JSON object,
no additional text or markdown formatting.
"""


def build_extraction_messages(
    case_presentation: str,
    correct_diagnosis: str,
    model_name: str,
    condition: str,
    reasoning_trace: str,
    include_gold: bool = True,
) -> list[dict]:
    """Build the messages array for a graph extraction API call.

    Args:
        include_gold: If False, omits the correct diagnosis from the prompt
                      to prevent potential extraction bias.
    """
    if include_gold:
        user_content = GRAPH_EXTRACTION_USER.format(
            case_presentation=case_presentation,
            correct_diagnosis=correct_diagnosis,
            model_name=model_name,
            condition=condition,
            reasoning_trace=reasoning_trace,
        )
    else:
        user_content = GRAPH_EXTRACTION_USER_NO_GOLD.format(
            case_presentation=case_presentation,
            model_name=model_name,
            condition=condition,
            reasoning_trace=reasoning_trace,
        )

    return [
        {"role": "system", "content": GRAPH_EXTRACTION_SYSTEM},
        {"role": "user", "content": user_content},
    ]
