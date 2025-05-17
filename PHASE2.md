# PHASE 2: Enhancing ML Operations with Containerization & Monitoring

## 1. Containerization
- [ ] **1.1 Dockerfile**
  - [ ] Dockerfile created and tested
  - [ ] Instructions for building and running the container
- [ ] **1.2 Environment Consistency**
  - [ ] All dependencies included in the container

## 2. Monitoring & Debugging

- [ ] **2.1 Debugging Practices**
  - [ ] Debugging tools used (e.g., pdb)
  - [ ] Example debugging scenarios and solutions

## 3. Profiling & Optimization
- **3.1 Profiling Scripts**
  - Profilers: Both cProfile and PyTorch Profiler were used. PyTorch profiler wasn't useful in our case because the model is a logisitc regression using numpy arrays and pandas data frames. cProfile generated usable results. Various cProfile functions can be called via the Makefile.
  - Profiling results and optimizations: The results as seen in snakeviz or in the tablular exports saved in reports/profiling, this revealed that we have very lean code. In searching for optimmizations we determined that most of the time running was dedicated to training and importing libraries. Some libraries imported and unused were removed, in addition, a line of code that referenced unused variables in our training algororithm was removed. 
    - tabular results stored in this folder [profiling](./reports/profiling/)
    - snakeviz output for training profile in this file [train_snakeviz.pdf](./docs/train_snakeviz.pdf)

## 4. Experiment Management & Tracking
- [ ] **4.1 Experiment Tracking Tools**
  - [ ] MLflow, Weights & Biases, or similar integrated
  - [ ] Logging of metrics, parameters, and models
  - [ ] Instructions for visualizing and comparing runs

## 5. Application & Experiment Logging
- [ ] **5.1 Logging Setup**
  - [ ] logger and/or rich integrated
  - [ ] Example log entries and their meaning

## 6. Configuration Management
- [ ] **6.1 Hydra or Similar**
  - [ ] Configuration files created
  - [ ] Example of running experiments with different configs

## 7. Documentation & Repository Updates
- [ ] **7.1 Updated README**
  - [ ] Instructions for all new tools and processes
  - [ ] All scripts and configs included in repo

---

> **Checklist:** Use this as a guide for documenting your Phase 2 deliverables. Focus on operational robustness, reproducibility, and clear instructions for all tools and processes.
