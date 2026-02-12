# L39: Validator Agent - Risk & Compliance

Enterprise-grade compliance validation system for VAIA with rule-based risk assessment and audit trails.

## 🎯 Features

- **Compliance Rule Engine**: Domain-specific rules (healthcare, financial, data privacy)
- **Hybrid Validation**: Pattern matching + LLM classification
- **Risk Scoring**: Aggregate severity scoring with configurable thresholds
- **Immutable Audit Trail**: PostgreSQL persistence for regulatory compliance
- **Real-time Dashboard**: Live compliance statistics and violation tracking
- **Two-Stage Validation**: Factual consistency (L38) + Compliance (L39)

## 🔑 API Key (required)

The validator uses **Google Gemini** for factual consistency and LLM-based compliance checks. Set your key before starting:

```bash
export GOOGLE_API_KEY=your_google_api_key_here
```

Get a key at [Google AI Studio](https://aistudio.google.com/apikey). For Docker, pass the same variable when starting (e.g. `GOOGLE_API_KEY=xxx ./start.sh` or use a `.env` file).

## 🚀 Quick Start

```bash
# Build system
./build.sh

# Set your Gemini/Google API key (see above)
export GOOGLE_API_KEY=your_key

# Start all services
./start.sh

# Run tests
./test.sh

# Stop system
./stop.sh
```

## 📊 Endpoints

- **Dashboard**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🛠️ API Usage

```bash
# Validate content
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{"id": "1", "content": "...", "source": "..."}],
    "query": "...",
    "domain": "healthcare"
  }'

# Get statistics
curl http://localhost:8000/stats

# Get audit trail
curl http://localhost:8000/audit-trail?limit=50
```

## 📝 Compliance Rules

Rules are defined in `rules/compliance_rules.yaml`:

```yaml
rules:
  - id: "RULE_ID"
    domain: "healthcare|financial|all"
    description: "Rule description"
    severity: 0-100
    pattern: "regex_pattern"  # Optional
    requires_classification: true/false
    action: "block|warn|filter"
```

## 🏗️ Architecture

```
ValidatorAgent
├── FactualConsistencyChecker (L38)
└── ComplianceEngine (L39)
    ├── RuleLoader
    ├── RuleEvaluator
    ├── RiskScorer
    └── AuditLogger
```

## 🔐 Security

- All validation decisions logged to immutable audit trail
- Redis caching with TTL for performance
- Circuit breaker prevents cascade failures
- WebSocket for real-time compliance monitoring

## 📈 Integration with L40

Emits `ComplianceViolationEvent` for self-healing feedback loops.

## 🎓 Learning Objectives

- Implement compliance-as-code frameworks
- Build hybrid validation (deterministic + LLM)
- Design immutable audit trails
- Scale rule evaluation with caching
- Integrate regulatory requirements into AI systems

---

Built for the VAIA 90-Lesson Curriculum • Module 5: Advanced RAG Orchestration
