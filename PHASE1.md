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
- **2.1 Repository Setup**
  - Github Repository Link: https://github.com/Davis24/Breast-Cancer-Classification---SE489 
  - Cookiecutter-Data-Science Template Used: https://cookiecutter-data-science.drivendata.org/ 
- **2.2 Environment Setup**
  - Each Member used a Python Virtual Environment. 
  - The requirements.txt can be found in: https://github.com/Davis24/Breast-Cancer-Classification---SE489/blob/main/requirements.txt 

## 3. Version Control & Collaboration
- **3.1 Git Usage**
  - We used regular commits with clear messages as can be seen on our main branch here https://github.com/Davis24/Breast-Cancer-Classification---SE489/commits/main/
  - We used branching and then adhered to a pull request template Template: https://github.com/Davis24/Breast-Cancer-Classification---SE489/wiki/Pull-Request-Template
- **3.2 Team Collaboration**
  - Roles on our team are not fixed, they are fluid as we each adapt to project needs. Generally speaking the following is a description of our roles: 
    - Megan Davis:
      - Setup the github project, github wiki, and standards for using github. 
      - Worked on project documentaton
      - Created the KNN Classification for testing purposes.
      - Organized code into callable functions 
      - Reviewed and approved pull requests
    - Nikki Frank: 
      - Created a logistic regression model in order to verify it's performance on our dataset
      - Worked on project documentation
      - Established our preliminary dvc pipeline
      - Reviewed and approved pull requests
    - Abe Berkley-Vigil
      - Met with the professor many times to ensure our group is on the right path
      - Explored to possibility of using an SVM model for our dataset
      - Worked on project documentation
      - Reviewed and approved pull requests

  - Code reviews and merge conflict resolution: The team performed robust Code Reviews and Merge Conflict resolutions. Each PR requires certain criteria to be followed and approved by the two other team members. 

## 4. Data Handling
- **4.1 Data Preparation**
  - Scripts COMING SOON!
- **4.2 Data Documentation**
  - Data preparation was minimal, as the dataset was verified to have been already standardized. The target was extracted and converted to 1s and 0s instead of Bs and Ms. The ID feature was removed from the dataset as it had no bearing on a sample's diagnosis.

## 5. Model Training
- **5.1 Training Infrastructure**
  - Training environment setup: In this phase of the project we were still running code on our local machines without a container. The model was first run on an Apple M1 Pro CPU in a python virtual environment managed by conda.
- **5.2 Initial Training & Evaluation**
  - Baseline model results: 
    - Accuracy = 94%
    - Confusion Matrix:  [[73   2] [ 5 34]] demonstrating that they model generates 5 false negatives and 2 false positives on the test data. The 5 false negatives are concerning and we may want to re-examine the model in the future to reduce them. False negatives are very harmful in the case of cancer detection.
  - Evaluation metrics: Accruacy and Confusion Matrix

## 6. Documentation & Reporting
- **6.1 Project README**
  - [README.md](./README.md)
- **6.2 Code Documentation**
  - Docstrings, inline comments, code style (ruff), type checking (mypy), Makefile docs LINKS COMING SOON!

## 7. Phase1 Reflections

Findings:

The model we decided to use was a logistic regression model. We felt that this model had a strong performance without overfitting the data. Using a train/test split of 0.2 we achieved accuracy .94 on the out of sample data. We had a precision of .94 and a recall of 0.97 for predicting benign and 0.94 precision and 0.87 recall for predicting malignant. We used a 30 factor model with our 5 strongest factors being radius_mean, concavity_worst, texture_se, symmetry_worst, and radius_worst.

For version controlling we used git. We found that using pull requests was the best way to direct our team members to what aspects of the project they should be working on. We also established a system of enabling two, non-author team members to review the code, but only requiring one of them to sign off on the pull request for merging. We achieved our pull requests by creating our own branches and merging them into main via rebasing. This allowed us to maintain a linear history of commits. This was feasible as our pull requests were on independent features of the project and thus we would not encounter any conflicts during the merge process.

For our research process we created jupyter notebooks to easily perform visualizations and quickly produce model results. Ultimately, to make our code better fit into an mlops environment we had to transform our jupyter notebooks into separate .py files for each process of the model which would allow us to produce doc strings, makefiles, and allow for a user to test the separate functions within our code.

Challenges Encountered:

The first challenge we encountered was selecting a proper dataset. Our criteria for a dataset was one that seemed to have been used frequently to validate it as a strong dataset. Additionally, we wanted a dataset that was mostly ready to use and did not require a lot of preprocessing given that our evaluation is more focused on the mlops pipeline surrounding the model. Thus, we did not want to spend extra effort in getting the data ready for modelling. Ultimately, we settled on the breast cancer dataset because it had been used many times on Kaggle, had sufficient data, and was already very clean to use dataset.

The second challenge we encountered was deciding which model among our SVM, KNN, and Logistic regression model we should choose. We decided on the logistic regression model because it seemed to be easily interpretable and had the most reasonable performance. It also seemed conducive to other future parts of the mlops pipeline that we will be using later in the project.

The other challenge we had was getting our entire team up to speed on how to use git and dvc properly. Regarding git, the various members of our team had different levels of

experience with it. Megan being the most experienced took the lead in setting up the repository. She also patiently guided the rest of the team members through the git repository, commands, pull requests, and merging. This allowed all the team members to use git effectively and merge the branches that contained each of our ML models into main.

Nikki faced an issue regarding setting up DVC on her google drive repository. Google drive has adopted new authentication protocols that were preventing the group from properly being able to use DVC. Nikki was able to work around this authentication issue by creating a service account that allowed our group to push and pull to that repo. She utilized DVC commands to workaround the authentication issue via using the service account’s key. Megan also provided a link to a DVC tutorial to help get the team familiar with DVC commands.

Areas For Improvement:

We seem to be more successful at predicting benign than malignant. This could be an issue given that we would want to more predictive of the cancer than not. Thus, we could possibly change our thresholds, change our sampling ratio, or our threshold for predicting cancer to target more effectively predicting cancer in the out of sample data set.

Another area for improvement is that we settled on one model when we could have created a function that allowed for multiple models to be considered. We could have used all three of our models and created a function that takes the model type as a parameter and a dictionary of hyperparameters as another parameter. Thus, we could have a more dynamic prediction environment and allow for greater iteration over hyperparameters. This would give us a wider breadth for using hydra and logging the differences between each model and its associated hyperparameters.