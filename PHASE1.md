# PHASE 1: Project Design & Model Development

## 1. Project Proposal
- **1.1 Project Scope and Objectives**
  - Problem statement: Radiologists study for years to be able to accurately identify whether cancer is present in the images they gather, but they can still make errors. To improve their accuracy, we need to deploy a sustainable and reproducible machine learning model that can detect cancer in breast screenings and continuously improve its predictions as it’s presented with new data. This tool must generate the same predictions on various machines. 
  - Project objectives and expected impact: Build a successful supervised binary classification model for detecting breast cancer. This will allow doctors to have a “second set of eyes” when identifying patients with breast cancer. 
  - Success metrics: To build a successful production model we want to adhere to two primary metrics: (1) the accuracy of the model, (2) the operational supportability of the model, and (3) the reproducibility and repeatability of the product. 
    - Accuracy: When we consider the fact that we are identifying lumps as either being malignant (cancer) or benign, either classification being incorrect would lead to a bad result for the individual involved. If a patient with cancer was classified as benign then they would not be receiving the treatment they need. If a patient without cancer was classified as with cancer, then they would be receiving treatment they did not need.  Either case is not acceptable. 
    - Operational Support: While building a model may garner most of the attention the end goal is always for it to be running successfully, with minimal interference in a production environment. To achieve this the model should be fully automated, documented, and be able to be easily supported from an operational team i.e. graceful failing, recovery, and detecting model drift.  
    - Reproducibility and Repeatability: The model must be able to reproduce the same classification from the same inputs on different machines, in addition, it must be able to repeat the same classification on one machine. 

  - Project Description: In SE489, we’ve been exploring a variety of productionalization techniques for the data science modeling life cycle. In this project, we seek to employ those techniques in order to address this project’s problem statement. According to the American Cancer Society, "breast cancer is the most common cancer in women in the United States, except for skin cancers. It accounts for about 30% (or 1 in 3) of all new female cancers each year.” We are going to deploy a model that can assist radiologists in their determination of whether breast cancer is present in breast cancer screening images.  
  
    A dataset that can reliably provide the base for such a model is key. We will use the data set found here, https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data. It contains 569 samples, 30 features, and 1 target. The target determines whether a sample is malignant or benign. The features describe the tumor or growth that was imaged. None of the raw image data is present in this dataset, instead, the important features of the original images have been recorded in a sample feature data. Features include things like “compactness”, “concavity”, “radius”, etc. This dataset is almost perfect, we would like to have more samples, but we are content to use what we have.   
    
    In order to create an optimal model for our deployment, we intend to test and train scikit learn’s pre-built models like KNN, SVM, and logistic regression. We will run experiments cross-referencing each model’s performance on differing hyperparameters and evaluate them by measuring their accuracy and loss. We can ensure we are not overfitting any models by watching the loss. The model we use will be the best performer without overfitting.  
    
    As a team, after all phased releases are completed, we will ensure that our model deployment is reproducible and sustainable (retrainable) by adhering to the following:   
      - Carefully maintaining the package requirements of the environment, via a package manager like conda. 
      - Organizing the project’s repository in a way that adheres to machine learning deployment best practices with the use of a cookiecutter template(s).   
      - Version controlling our code during development by using git technology and a remote github repository as the “source of truth.”  
      - Ensuring we follow PEP8 style guidelines by using linters and formatters like ruff.  
      - Controlling the various versions of our model, data, and features through data version control via a tools like DVC. This would allow us to pair feature selection, train test splits, and other data manipulations with differing models.  
      - Maintaining an easy-to-follow recipe on how to reproduce the model we create and the various experimental hyperparameters that were used via a tool like hydra.  
      - Curating a containerized version of our project that can be reliably run on any machine using tools like docker.  
      - Including key function calls (like the ability to retrain) in an organized Make file. 

 

By adhering to the best practices in an MLOps production lifecycle we will develop our reproducible and sustainable breast cancer classifier. 

- **1.2 Selection of Data**
  - Dataset(s) chosen and justification: After reviewing numerous datasets on Kaggle we decided upon the Breast Cancer Data set (https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data). We chose this data set because it has usability score of 10.0 with 542 upvotes, and nearly 409 projects created from it. The data has a reasonable split of 63% benign, and 37% malignant. It is a smaller dataset, with only 569 instances, however it has 30 possible features. Based on this we concluded it would be an ideal data set. 
  - Data source(s) and access method: The data size is only 124.57kB, and the last time the data was updated was 3 years ago. We will download the file and put it into our project structure where it will then be accessed by the model. 
  - Preprocessing steps: Depending on the model we select we may need to transform and scale the data. Given that we will be using a logistic regression, SVM, and KNN, min max scaling will be applied. Logistic regression may require us to transform our data depending on its distributional properties which will be identified through plots and histograms.  
- **1.3 Model Considerations**
  - Model architecture(s) considered: Logistic Regression, SVM, KNN
  - Rationale for model choice: Given the relatively small size of our data set, <600 data points, we will consider models that are robust to making classification predictions on small datasets. Robustness to small datasets is particularly important as we will be dividing our data into training and test sets. Additionally, given our small dataset will want to be as parsimonious as possible with regard to our factors, and a logistic regression will allow us to implement feature reduction techniques like a lasso. Support vector machines are effective when dealing with complex boundaries separating the two classes. 
  - Pre-built models used: 
    - KNN - sklearn.neighbors.KNeighborsClassifier 
    - SVM - sklearn.svm.SVC (SVC for classification) 
    - Logistic Regression - sklearn.linear_model.LogisticRegression 
- **1.4 Open-source Tools**
  - Third-party package(s) selected (not PyTorch or course-used tools):
    - Scikit learn - used for the pre-built models
    - Pandas and Numpy - both used for data manipulations and preprocessing
    - Seaborn - used for visualizations that will help us better understand our patterns in our dataset
    - matplotlib - used to create custom plots

## 2. Code Organization & Setup
- [ ] **2.1 Repository Setup**
  - [ ] GitHub repo created
  - [ ] Cookiecutter or similar structure used
- [ ] **2.2 Environment Setup**
  - [ ] Python virtual environment
  - [ ] requirements.txt or environment.yml
  - [ ] (Optional) Google Colab setup

## 3. Version Control & Collaboration
- [ ] **3.1 Git Usage**
  - [ ] Regular commits with clear messages
  - [ ] Branching and pull requests
- [ ] **3.2 Team Collaboration**
  - [ ] Roles assigned
  - [ ] Code reviews and merge conflict resolution

## 4. Data Handling
- [ ] **4.1 Data Preparation**
  - [ ] Cleaning, normalization, augmentation scripts
- [ ] **4.2 Data Documentation**
  - [ ] Description of data prep process

## 5. Model Training
- [ ] **5.1 Training Infrastructure**
  - [ ] Training environment setup (e.g., Colab, GPU)
- [ ] **5.2 Initial Training & Evaluation**
  - [ ] Baseline model results
  - [ ] Evaluation metrics

## 6. Documentation & Reporting
- [ ] **6.1 Project README**
  - [ ] Overview, setup, replication steps, dependencies, team contributions
- [ ] **6.2 Code Documentation**
  - [ ] Docstrings, inline comments, code style (ruff), type checking (mypy), Makefile docs

---

> **Checklist:** Use this as a guide. Not all items are required, but thorough documentation and reproducibility are expected.
