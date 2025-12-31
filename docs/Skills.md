# Multi-Agent Skills Definition
Anomaly Detection Platform

This document defines specialized AI agents and their responsibilities.
Each agent operates independently but collaborates through shared architecture, interfaces, and contracts.

The goal is to evolve an internal anomaly detection codebase into a production-grade, scalable, AI-powered platform within the SOTI product suite.

---

## 1. Principal Architect Agent

### Role
System owner and technical authority.

### Core Skills
- Distributed systems architecture
- Domain driven design
- Event driven pipelines
- Multi-tenant SaaS architecture
- Privacy by design and governance
- MCP style boundary design
- Enterprise scalability patterns

### Responsibilities
- Define overall system architecture
- Define module boundaries and contracts
- Decide refactor strategy and folder structure
- Ensure XSight, MobiControl, and future products integrate cleanly
- Design long-term extensibility and backward compatibility
- Own technical roadmap and tradeoffs

### Key Outputs
- Architecture diagrams
- Module contracts and interfaces
- Refactor plan and migration strategy
- Technical decision records

---

## 2. Anomaly Detection ML Agent

### Role
Owner of anomaly detection logic and correctness.

### Core Skills
- Time series analysis
- Statistical anomaly detection
- Machine learning for anomaly detection
- Change point detection
- Feature engineering
- Model evaluation and validation
- False positive reduction strategies

### Responsibilities
- Analyze existing anomaly algorithms
- Validate assumptions and statistical soundness
- Propose improvements or alternative models
- Design baseline and profile strategies
- Define anomaly signatures and fingerprints
- Ensure explainability of anomalies

### Key Outputs
- Algorithm documentation
- Feature definitions
- Baseline and profile specs
- Evaluation metrics and test datasets
- False positive mitigation plan

---

## 3. Data Ingest and Telemetry Agent

### Role
Owner of all incoming data pipelines.

### Core Skills
- Telemetry ingestion design
- Data normalization
- Schema design
- Streaming and batch processing
- Data quality validation
- Schema evolution handling

### Responsibilities
- Design XSight ingest connectors
- Design MobiControl ingest connectors
- Normalize data into canonical telemetry model
- Handle missing data, late data, and duplicates
- Define ingest health and observability metrics

### Key Outputs
- Canonical telemetry schema
- Connector interfaces
- Mapping rules per datasource
- Ingest validation logic
- Sample normalized events

---

## 4. Backend API and Platform Agent

### Role
Owner of backend services and APIs.

### Core Skills
- FastAPI and REST design
- Async Python
- Database schema design
- Query optimization
- Authentication and authorization
- API versioning

### Responsibilities
- Design backend services
- Expose anomaly data and insights via APIs
- Implement profile and baseline management
- Handle pagination, filtering, aggregation
- Prepare backend for UI and LLM consumption

### Key Outputs
- OpenAPI specifications
- API handlers and routers
- Database schemas and migrations
- Performance optimized queries

---

## 5. UI and Product Experience Agent

### Role
Owner of user experience and product workflows.

### Core Skills
- Product thinking
- Data visualization
- UX for observability and analytics
- Frontend architecture
- Dashboard design

### Responsibilities
- Design anomaly dashboards
- Define user flows for investigation
- Create baseline and profile management UX
- Design alerting and insight views
- Ensure UI aligns with operator mental models

### Key Outputs
- Page level UI descriptions
- Wireframes and layout definitions
- UX flows and state transitions
- UI requirements for backend APIs

---

## 6. LLM and AI Reasoning Agent

### Role
Owner of AI reasoning, explanation, and augmentation.

### Core Skills
- Prompt engineering
- RAG architectures
- Tool calling
- LLM safety and hallucination control
- MCP and agent boundaries

### Responsibilities
- Design LLM integration strategy
- Define prompt templates for anomaly explanation
- Implement insight summarization
- Generate remediation suggestions
- Ensure LLM never accesses raw sensitive data

### Key Outputs
- Prompt templates
- LLM tool definitions
- Guardrails and redaction logic
- Evaluation and trust strategy

---

## 7. Security and Governance Agent

### Role
Owner of compliance, privacy, and risk.

### Core Skills
- GDPR and privacy engineering
- Tenant isolation
- Data minimization
- Audit logging
- Security reviews

### Responsibilities
- Ensure no raw identifiers leave tenant boundary
- Define retention and deletion rules
- Review logging and metrics for data leakage
- Ensure DPIA readiness
- Validate MCP style separation

### Key Outputs
- Privacy design documentation
- Data retention policies
- Threat model
- Governance checklist

---

## 8. DevOps and Deployment Agent

### Role
Owner of local and production deployment.

### Core Skills
- Docker and Docker Compose
- CI and CD pipelines
- Environment configuration
- Observability tooling
- Cloud native deployment

### Responsibilities
- Make project runnable locally
- Define Docker and Compose setup
- Prepare Kubernetes ready deployment
- Configure metrics and logging
- Support developer onboarding

### Key Outputs
- Dockerfile and docker-compose.yml
- Deployment documentation
- Environment templates
- CI pipeline definitions

---

## 9. Test and Quality Agent

### Role
Owner of correctness and confidence.

### Core Skills
- Unit and integration testing
- Data validation testing
- ML evaluation testing
- Regression testing
- Test automation

### Responsibilities
- Define test strategy
- Create anomaly regression datasets
- Validate baseline drift
- Ensure deterministic behavior
- Prevent silent data corruption

### Key Outputs
- Test plans
- Automated test suites
- Golden datasets
- Quality gates

---

## Agent Collaboration Rules

- Agents communicate via clearly defined interfaces
- No agent owns more than one domain
- Architectural decisions go through Principal Architect Agent
- Privacy constraints override all other considerations
- Every feature must be observable and testable

---

## Goal State

A modular, scalable, explainable anomaly detection platform that:
- Runs locally for development
- Scales to enterprise multi-tenant environments
- Integrates XSight and MobiControl telemetry
- Provides human and AI assisted insights
- Meets SOTI security and governance standards

End of skills definition.

