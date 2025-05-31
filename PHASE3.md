# PHASE 3: Continuous Machine Learning (CML) & Deployment
#change for push
## 1. Continuous Integration & Testing
- [ ] **1.1 Unit Testing with pytest**
  - [ ] Test scripts for data processing, model training, and evaluation
  - [ ] Documentation of the testing process and example test cases
- [ ] **1.2 GitHub Actions Workflows**
  - [ ] CI workflows for running tests, DVC, code checks (e.g., ruff), Docker builds
  - [ ] Workflow YAML files included
- [ ] **1.3 Pre-commit Hooks**
  - [ ] Pre-commit config and setup instructions

## 2. Continuous Docker Building & CML
- [ ] **2.1 Docker Image Automation**
  - [ ] Automated Docker builds and pushes (GitHub Actions)
  - [ ] Dockerfile and build/push instructions for Docker Hub and GCP Artifact Registry
- [ ] **2.2 Continuous Machine Learning (CML)**
  - [ ] CML integration for automated model training on PRs
  - [ ] Example CML outputs (metrics, visualizations)
  - [ ] Setup and usage documentation

## 3. Deployment on Google Cloud Platform (GCP)
- [ ] **3.1 GCP Artifact Registry**
  - [ ] Steps for creating and pushing Docker images to GCP
- [ ] **3.2 Custom Training Job on GCP**
  - [ ] Vertex AI/Compute Engine job setup and documentation
  - [ ] Data storage in GCP bucket
- [ ] **3.3 Deploying API with FastAPI & GCP Cloud Functions**
  - [ ] FastAPI app for model predictions
  - [ ] Deployment steps and API testing instructions
- [ ] **3.4 Dockerize & Deploy Model with GCP Cloud Run**
  - [ ] Containerization and deployment steps
  - [ ] Testing and result documentation
- [ ] **3.5 Interactive UI Deployment**
  - [ ] Streamlit or Gradio app for model demonstration
  - [ ] Deployment on Hugging Face platform
  - [ ] Integration of UI deployment into GitHub Actions workflow
  - [ ] Screenshots and usage examples

## 4. Documentation & Repository Updates
- [ ] **4.1 Comprehensive README**
  - [ ] Setup, usage, and documentation for all CI/CD, CML, and deployment steps
  - [ ] Screenshots and results of deployments
- [ ] **4.2 Resource Cleanup Reminder**
  - [ ] Checklist for removing GCP resources to avoid charges

---

> **Checklist:** Use this as a guide for documenting your Phase 3 deliverables. Focus on automation, deployment, and clear, reproducible instructions for all steps.
