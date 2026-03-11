# Email-Tagging-System
Differentiating mails and classify them under certain classes for companies and organisations with a large email data store

#  Customer-Isolated Email Tagging System
### Rule-Based Multi-Level Pattern Matching with LLM-Ready Architecture

A production-ready email auto-tagging system that classifies customer support emails into issue categories using a weighted pattern-matching classifier — with strict **per-customer tag isolation** to prevent cross-contamination between accounts.

---

##  Overview

This project implements an intelligent email tagging pipeline for customer support teams (built around Hiver). Given an incoming email (subject + body), the system:

1. **Isolates** the customer's specific tag vocabulary — no shared learning between accounts
2. **Scores** the email against 28+ pattern categories using multi-level keyword matching
3. **Predicts** the best-matching tag with a calibrated confidence score
4. **Validates** the prediction against the customer's allowed tag set
5. **Flags** low-confidence predictions for human review or LLM escalation

---

##  Architecture
```
Incoming Email (subject + body)
           │
           ▼
┌──────────────────────────┐
│  Customer Isolation Layer │  ← Lookup customer-specific tag vocab
│  (CustomerIsolatedTagging)│
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Feature Extraction       │  ← Exact phrases, primary & secondary keywords
│  (extract_advanced_feat.) │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Composite Scoring        │  ← 5-layer weighted scoring per valid tag
│  (calculate_tag_scores)   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Prediction + Confidence  │  ← Top tag by score + score-gap confidence
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Validation               │  ← Reject tags outside customer vocabulary
└────────────┬─────────────┘
             │
             ▼
   Structured Result
{ predicted_tag, confidence, is_valid, reasoning }
```

---

##  Features

- **Strict customer isolation** — each customer has a completely separate tag vocabulary; zero cross-contamination
- **28+ pattern categories** with primary, secondary, and special-case keyword scoring
- **Multi-level scoring** — exact phrases > primary keywords > secondary keywords > tag-specific rules
- **Calibrated confidence scores** based on absolute score magnitude and separation from the second-best candidate
- **Graceful fallback** to most-common-tag heuristic when pattern scores are inconclusive
- **LLM-ready prompt design** — the architecture is prepared for seamless GPT-4/Claude integration
- **Two datasets** — small (12 emails × 4 customers) and large (60 emails × 6 customers)
- **Error analysis** built in — identifies confusion patterns and low-confidence misclassifications

---

##  Datasets

| Dataset | Emails | Customers | Tags per Customer |
|---------|--------|-----------|-------------------|
| Small   | 12     | 4 (A–D)   | ~3 unique         |
| Large   | 60     | 6 (A–F)   | 10 unique         |

Each dataset contains `email_id`, `customer_id`, `subject`, `body`, and ground-truth `tag`.

---

##  Getting Started

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or local Jupyter/Python environment

### Installation
```bash
pip install pandas numpy
```

### Running

Execute the notebook cells in order, or run the script directly:
```bash
python Email_Tagging_Sys.py
```

---

##  Core Classes

### `CustomerIsolatedTaggingSystem`

Manages per-customer tag vocabularies and few-shot example storage.
```python
isolation_system = CustomerIsolatedTaggingSystem()
isolation_system.train(df_large)

# Validate a prediction against a customer's allowed tags
is_valid, msg = isolation_system.validate_prediction('CUST_A', 'billing_error')
# → False: 'billing_error' belongs to CUST_B, not CUST_A
```

| Method | Description |
|--------|-------------|
| `train(df)` | Builds per-customer tag sets and few-shot examples |
| `get_customer_context(customer_id)` | Returns few-shot prompt context for a customer |
| `validate_prediction(customer_id, tag)` | Checks tag is in customer's allowed vocabulary |

---

### `AdvancedEmailTagClassifier`

Classifies emails using composite multi-level pattern matching.
```python
classifier = AdvancedEmailTagClassifier(isolation_system)
result = classifier.classify('CUST_A', subject="Unable to access shared mailbox", body="...")
# → { predicted_tag: 'access_issue', confidence: 0.87, is_valid: True, reasoning: '...' }
```

| Method | Description |
|--------|-------------|
| `extract_advanced_features(text)` | Detects exact phrases, primary/secondary keyword matches |
| `calculate_tag_scores(text, valid_tags)` | Scores all valid tags via 5-layer weighted rules |
| `classify(customer_id, subject, body)` | Full pipeline — returns prediction dict |

---

##  Customer Isolation

Three layers ensure zero cross-customer tag leakage:

1. **Data Segregation** — separate dictionaries per customer for tags and examples
2. **Training Isolation** — each customer trained independently, no shared learning
3. **Validation Layer** — every prediction checked against the customer's tag vocabulary before returning

**Isolation test results: 5/5 passed **

| Test | Customer | Tag | Expected | Result |
|------|----------|-----|----------|--------|
| 1 | CUST_A | `access_issue` |  Valid | PASS |
| 2 | CUST_A | `billing_error` |  Blocked (CUST_B tag) | PASS |
| 3 | CUST_B | `billing_error` |  Valid | PASS |
| 4 | CUST_C | `access_issue` |  Blocked (CUST_A tag) | PASS |
| 5 | CUST_D | `user_management` | Valid | PASS |

---

##  Scoring System

Each email is scored against every valid tag in the customer's vocabulary using 5 layers:
```
Layer 1 — Exact tag word match          → +5.0 × occurrences  (strongest signal)
Layer 2 — Exact multi-word phrase match → +4.0 × overlap
Layer 3 — Primary pattern category match→ +weight × 2.0
Layer 4 — Secondary pattern match       → +weight × 0.5
Layer 5 — Special-case rules            → +3.0–5.0 (tag-specific boosts)
```

**Confidence** = `min(max_score / 15.0, 0.95)`, boosted by +0.10 if score gap to 2nd-best > 5.0.

---

##  Pattern Categories (28+)

`billing` · `access` · `performance` · `automation` · `analytics` · `tagging` · `ui` · `notification` · `user_management` · `setup` · `feature_request` · `mail_merge` · `email_threading` · `search` · `draft` · `attachment` · `assignment` · `mobile` · `sync` · `export` · `sla` · `session` · `keyboard` · `bcc` · `outbox` · and more

---

##  Error Analysis

### Common Error Types

| Type | Share of Errors | Example | Root Cause |
|------|-----------------|---------|------------|
| Similar categories | ~40% | `workflow_issue` → `automation_bug` | Both involve rules/automation |
| Generic symptoms | ~30% | `ui_bug` → `editor_performance` | Low keyword overlap |
| Ambiguous language | ~30% | `setup_help` → `sla_issue` | Phrases like "help setting up SLAs" are ambiguous |

**Key insight:** 40–50% of errors have confidence < 0.5 — the system knows when it's uncertain.

---

##  LLM-Ready Prompt Design

The system is architected to route low-confidence cases to GPT-4 or Claude. The prompt structure is already built in:
```
Valid tags for CUST_A: [access_issue, workflow_issue, ...]

Example 1:
Subject: Unable to access shared mailbox
Body: Getting permission denied...
Tag: access_issue

[2–3 more few-shot examples]

Classify this email:
Subject: [NEW SUBJECT]
Body: [NEW BODY]
Tag:
```

---

##  Production Improvements

Three high-impact enhancements are documented in the codebase:

1. **Hybrid LLM + Embeddings** — Use pattern matching for high-confidence cases; escalate low-confidence ones to GPT-4 with semantic embeddings. Expected: 50% cost reduction vs. pure LLM, +10–15% accuracy.

2. **Active Learning Pipeline** — Flag predictions with confidence < 0.6 for weekly human review. Incrementally retrain and adjust pattern weights from corrections. Expected: +5–10% accuracy after 1 month.

3. **Multi-Model Ensemble** — Combine pattern matcher + fine-tuned BERT + GPT-4 + gradient boosting with weighted voting. High model agreement boosts confidence; low agreement reduces it. Expected: +8–12% accuracy, better-calibrated confidence.

---

##  Tech Stack

| Component | Technology |
|-----------|------------|
| Data handling | `pandas`, `numpy` |
| Pattern matching | `re` (regex), custom weighted scoring |
| Customer isolation | `collections.defaultdict` |
| LLM integration (future) | OpenAI GPT-4 / Anthropic Claude |
| Environment | Google Colab / Python 3.8+ |

---
