"""
L43: Seed the ChromaDB collection with enterprise knowledge documents.
Run once before starting the API server.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.retriever import RetrieverAgent

DOCUMENTS = [
    "The company's Q3 2025 revenue increased by 18% year-over-year, driven by strong performance in the cloud services division. Enterprise contracts grew 24%, representing 63% of total revenue.",
    "Our AI deployment guidelines require all production models to pass security review and bias assessment before release. Models must achieve >95% accuracy on the internal benchmark suite.",
    "The incident response protocol mandates a 15-minute initial triage window for P1 incidents. The on-call engineer must notify the CTO within 30 minutes of P1 classification.",
    "Employee benefits include comprehensive health insurance, 20 days PTO per year, and a $5,000 annual learning budget. Remote work is permitted for eligible roles with manager approval.",
    "The data retention policy requires customer data to be deleted within 90 days of account closure. Backups are retained for 7 years for compliance with financial regulations.",
    "Our microservices architecture consists of 47 services deployed across three availability zones. Average service-to-service latency is 12ms at p99. We use gRPC for internal communication.",
    "The product roadmap for 2026 includes: autonomous agent orchestration framework (Q1), multimodal document processing (Q2), real-time collaboration features (Q3), and global CDN expansion (Q4).",
    "Security audit results from October 2025 identified 3 high-severity findings, all of which were remediated within the 30-day SLA. The next audit is scheduled for April 2026.",
    "Our machine learning infrastructure runs on a cluster of 200 A100 GPUs. Training jobs are scheduled via a priority queue system. Average GPU utilization is 78% during business hours.",
    "Customer success metrics: Net Promoter Score is 67 (industry average: 45), customer churn rate is 2.1% monthly, average contract value increased to $180K in 2025.",
    "The engineering team follows a 2-week sprint cadence with daily standups. Code review requires at least 2 approvals. All features must include unit tests with >80% coverage.",
    "Compliance certifications held: SOC 2 Type II, ISO 27001, GDPR compliant, HIPAA BAA available for healthcare customers. Annual penetration testing conducted by third party.",
    "Vector database performance benchmarks: ChromaDB handles 10M embeddings with p99 query latency of 45ms. We shard collections at 2M embeddings for optimal performance.",
    "The Agentic RAG system uses a four-stage pipeline: query planning, multi-source retrieval, validation with self-correction, and grounded synthesis using large language models.",
    "Enterprise AI governance framework includes: model cards for all production models, quarterly bias audits, explainability reports for high-stakes decisions, and human oversight for critical workflows.",
]

SOURCES = [
    "financial_report_q3_2025.pdf",
    "ai_deployment_guidelines.md",
    "incident_response_protocol.md",
    "hr_benefits_handbook.pdf",
    "data_retention_policy.md",
    "infrastructure_overview.md",
    "product_roadmap_2026.pdf",
    "security_audit_2025.md",
    "ml_infrastructure_guide.md",
    "customer_success_metrics.md",
    "engineering_handbook.md",
    "compliance_certifications.md",
    "vector_db_benchmarks.md",
    "agentic_rag_documentation.md",
    "ai_governance_framework.md",
]

if __name__ == "__main__":
    print("Seeding ChromaDB knowledge base...")
    retriever = RetrieverAgent("enterprise_kb")
    # Check if already seeded
    try:
        count = retriever._collection.count()
        if count >= len(DOCUMENTS):
            print(f"Collection already has {count} documents. Skipping seed.")
            sys.exit(0)
    except Exception:
        pass
    n = retriever.ingest(DOCUMENTS, SOURCES)
    print(f"✓ Ingested {n} documents into 'enterprise_kb' collection")
    print(f"  Collection count: {retriever._collection.count()}")
