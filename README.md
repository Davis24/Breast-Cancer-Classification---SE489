# Breast Cancer Classification

## 1. Team Information
- [ ] Team Name: Team Python
- [ ] Team Members: (Megan Davis, MDAVI166@depaul.edu), (Nikki Frank, NFRANK6@depaul.edu), (Abe Berkley-Vigil, ABERKLEY@depaul.edu)
- [ ] Course & Section: SE 489 Section 930, 910

## 2. Project Overview
- [ ] Summary: Radiologists study for years to be able to accurately identify whether cancer is present in the images they gather, but they can still make errors. To improve their accuracy, we need to deploy a sustainable and reproducible machine learning model that can detect cancer in breast screenings and continuously improve its predictions as it’s presented with new data. This tool must generate the same predictions on various machines.
- [ ] Problem statement and motivation
- [ ] Main objectives

## 3. Project Architecture Diagram
- [ ] Insert or link to your architecture diagram (e.g., draw.io, PNG, etc.)

## 4. Phase Deliverables
- [ ] [PHASE1.md](./PHASE1.md): Project Design & Model Development
- [ ] [PHASE2.md](./PHASE2.md): Enhancing ML Operations
- [ ] [PHASE3.md](./PHASE3.md): Continuous ML & Deployment

## 5. Setup Instructions
- [ ] How to set up the environment (conda/pip, requirements.txt, Docker, etc.)
- [ ] How to run the code and reproduce results

## 6. Contribution Summary
- [ ] Briefly describe each team member's contributions

## 7. References
- [ ] List of datasets, frameworks, and major third-party tools used

---

> **Tip:** Keep this README updated as your project evolves. Link to each phase deliverable and update the architecture diagram as your pipeline matures.


<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

SE 489 project - breast cancer classification.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         beast_cancer_classification and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── beast_cancer_classification   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes beast_cancer_classification a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

