# PHASE 2: Enhancing ML Operations with Containerization & Monitoring

## 1. Containerization
- **1.1 Dockerfile**
Docker documentation can be found in the `/docker/README.md` file, however we will duplicate some of the documentation here for ease of use.
  - While we highly recommend you follow the [instructions](https://docs.docker.com/get-started/get-docker/) provided by Docker itself. However we will detail some brief instructions below. 
    - Install Docker GUI for your respective operating system. We recommend the GUI since it is easy to use.   The installation .exe can be found [here](https://docs.docker.com/get-started/get-docker/).
    - Follow the .exe instructions. 
    - Once installed restart your machine.
    - `VSCode` - If you wish to include Docker information in your VSCode you can install the [VScode Docker extension](https://code.visualstudio.com/docs/containers/overview).
  - Commands:
    - Building a docker image:
      - Run `docker build --no-cache -f docker/<docker-file-name> . -t <tag-name>:latest`
      - Example: `docker build --no-cache -f train.dockerfile . -t train:latest`
    - Be patient this should take around 2-3 minutes to fully build.
    - Running:
      - Run `docker run --name <name> <tag-name>:latest`  
      - Example: `docker run --name exp1 train:latest`
  - Note: Traditionally we would expect to have our train and predict dockerfiles to be smaller than one another. However, breast-cancer-classification was built to act as a python module. Due to this we cannot easily separate out particular scripts or package requirements. As such both train and predict have the same size at ~1.4GB.
- **1.2 Environment Consistency**
  - All dependencies are built using the `requirements.txt` document. 

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
- **4.1 Experiment Tracking Tools**
  - Weights and Biases was integrated into our project for the purposes of logistic regression model experimentation, and hyperparameter optimization.
  - Logging of the various experiments in the form of sweeps can be found in the wandb project called Breast-Cancer-Classification---SE489-experiments_wandb. We determined our logisitc regression model performs best with the following hyperparameters: 
      -  C=1, solver = lbfgs, and max_iterations = 300. 
  - We have implemented those hyperparameters into our production training script train.py
    - The full report with all 32 different scenarios included is in this link: https://api.wandb.ai/links/nfrank6-depaul-university/rim95641
    - The best scenarios report can be found at this link: https://api.wandb.ai/links/nfrank6-depaul-university/c3zgwb82
    - The best run based on test accuracy and f1 score is saved in this repo path nfrank6-depaul-university/Breast-Cancer-Classification---SE489-experiments_wandb/ubv3xqel. It can be accessed with this link if granted access (currently not paying for higher subcription tier that will allow access to others) https://wandb.ai/nfrank6-depaul-university/Breast-Cancer-Classification---SE489-experiments_wandb/runs/ubv3xqel/overview
  - Instructions for visualizing and comparing runs: 
      1. You need to connect to my wandb account Run init.py found here: [init](./experiments/wandb/) and select 2 (for existing account). Then insert the API key: cd7e0f9a3007c63400085c96e0ce74052fdfcd2d. If you choose to create your own account, first go online and create your wandb account at wandb.ai, then proceed with the steps above and insert your own api key. 
      2. Create a new sweep in wandb by running in the terminal: wandb sweep experiments/wandb/sweep.yaml
      3. The command above will generate a run command which you can copy and then run. The command will look something like the following, but it's custom to each new sweep you initialize: wandb agent nfrank6-depaul-university/Breast-Cancer-Classification—SE489/ax3kf4aa
      4. Go to your connected project at wandb.ai, view your sweeps, then explore the differences between runs. There you can create reports of your own.


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
