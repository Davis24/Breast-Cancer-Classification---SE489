# PHASE 2: Enhancing ML Operations with Containerization & Monitoring



## 1. Containerization

For containerization we moved forward with using Docker. This allows us to create self-contained environments to execute the Breast Cancer Classification code without requiring everyone on the team to setup their own environment. Instead we built docker images for train and predict, that can be excecuted by docker CLIs. See the documentation below.

- **1.1 Dockerfile**
More detailed Docker documentation can be found in the `/docker/README.md` file, however we will duplicate some of the documentation here for ease of use.
  - we highly recommend you follow the [instructions](https://docs.docker.com/get-started/get-docker/) provided by Docker itself. However we will detail some brief instructions below. 
    - Install Docker GUI for your respective operating system. We recommend the GUI since it is easy to use.  The installation .exe can be found [here](https://docs.docker.com/get-started/get-docker/).
    - Follow the .exe instructions. There is no reason to deviate from the standard instructions.
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

After reviewing the tool recommended in the documentation, Prometheus, it was decided that this tool was heavy handed for the simplicity of our model. Instead the model performance is tracked by the logging provided within the other sections.

- **2.1 Debugging Practices**
  - When it came to debugging our code we utilized the pdb as well as tools within Visual Studio Code like Python Debugger Extension and Data Wrangler. Below we will detail some scenarios in which these tools were used.
  - `Data Wrangler`: Often it can be difficult to visual large dataframes, this is because the traditional print tools within python are very limited. They require specific formatting and don't allow for any form of manipulation. Instead we can use the Data Wrangler plugin by Microsoft. It allows the dataframe to be opened in a separate panel within VS Code, from here the data can be sorted, filtered, and so on. This allows us to quickly identify if there are any issues within dataframe at each step.
  - `Python Debugger Extension` & `pdb`: This two tools are very similar in the way they function. We mention them together because they both use the exact same type of interactions. Breakpoints and step throughs. The primary difference being the extension allows for the placement of breakpoints through the GUI in VSCode, whereas pdb requires the implementation of code snippets into the python file to be hit during execution. In our particular instance we primarily worked with the VScode debugger extension.
    - An example of our usage with the python debugger extension is early on during the construction of the breast_cancer_classification module we were refactoring it from a .ipynb. As we were splitting code out in proper methods we needed to ensure the step through of the process was following the same path as th .ipynb. By using break points with step through we were able to ascertain that the code steps were following the same process as those in the .ipynb. 
    - There are other areas in which this could be applied such as difficult to identify bugs or crashes however we did not experience any of those during the creation of our model.

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
- **5.1 Logging Setup**
  - Logging is used to track the progress and help with debugging  of code. It is also used for error tracking. Thus, our team found it important that we both track logging via the console and storing the logs. 
- We produced logs for three files, dataset.py, predict.py, and train.py. The file dataset.py handles loading and preprocessing the data for training. The train.py file works with hydra to take in the optimal model parameters, splits, fits,  and trains on the data. 
Our approach to logging was to be fairly verbose. For our team it was important to be able to pinpoint where our code was at in the console’s logs. This helps with debugging and finding issues when they do occur. Thus, every function had some amount of logging. Most of our messages were info messages so we could monitor our progress throughout the running of the code, so logging with info acted as a checkpoint for our code. We also use success messages or failure messages to know if our code had completed properly.
- Our team used other logging messages when relevant in our code like warning or error messages. For example in our load_data function in the dataset.py file, we print a warning if there are any missing values in our dataset. This is important because model training or performance can be deteriorated or stopped due to missing values. Thus, we wanted to be aware of when this issue could arise. We used error messages in our main function in the train.py file. Since the train.py main function performs many subtasks such as creating the model, writing data csvs, and fitting the model we wanted to put that portion of the code in a try block and raise errors in an except block with a log recorded for such errors. 
We also logged information regarding general function and model performance. For example every function had a timing related log value. This is to track if any parts of the code are creating bottlenecks which could lead to future optimizations. Additionally, we print out information to our console like confusion matrices, data distribution, and model parameters. This allows the user to see the results of key sub-functions as the code is run. These values were printed to the console, but could not be written to a file. Thus, our logging process had two facets to it, console printing and file writing.
- We used rich handlers for a lot of our logging. This was done to make the logs easier to read. Rich handlers also allow you to color certain aspects of the logs. This can be used to highlight key pieces of information that may be relevant to the user. In our case, we print the dimensions of the training and testing sets in yellow and the label distributions in blue. Often machine learning models can appear to be highly accurate, but instead are often just predicting 1 class value for everything. Because seeing both the dimensions and label distributions we can understand how many factors we are modelling with, and if our data is imbalanced.
Finally we also created timestamped logs for our predict.py and dataset.py logs. This is so that for each time they are run, and new log can be created and associated with a specific time so we can track our experiments and their results.
Overall logging was used as a way to dynamically monitor our model runs and store the associated logs. 


## 6. Configuration Management
  **6.1 Hydra or Similar**
- Hydra is a tool used in MLOPs for configuring parameters in one’s project pipeline. Thus, I created three different yaml files to be used by hydra. The first file is a config.yaml file. This file drives much of the relevant information regarding both the inputs and the outputs of our team’s model. Our team is modelling breast cancer classification and as a result we are using a logistic regression for our model which means our parameters used in hydra, reflect those of a logistic regression. 
- We separated the config.yaml file into three areas, data, train, model. The section labelled Data hold the information regarding where the features and labels will be written to. It also holds the file path to the dataset for training the model. The train section sets whether debugging is defaulted to and whether the data should be scaled. It also contains the random state and test size. The model section pertains to the parameters used in the model and their associated values. For our model the only parameter we used was max_iter which controls the maximum amount of iterations for the fitting of the model. It also contains the path where the model will be dumped to a pkl file. 
We also created a file called hydra.yaml. This file helps indicate where files should be saved. Additionally it covers that log messages should be associated with levels and timestamps. It also covers the run setting for the experiments we could be running in hydra. We also create an overrides file that controls anything that may need to be overwritten like command line overrides. Since we don’t have any in our model this file is effectively blank.
- For our hydra experiment we decided to only run one model. This is because Nikki was able to use other MLOPs tool for optimization to find the optimal model. Thus, it would have been redundant to reproduce multiple experiments when the optimal model was found. Despite only making 1 model, the structure of the pipeline allows for many more models to be made. Specifically, the config.yaml file could be rewritten to test a broader swath of parameter values. Additionally, we could have tested more logistic regression parameters like the C value which controls the regularization parameter. We also made the random state a parameter that can be maintained or changed so that our experiment can be replicable by other users of this overall pipeline, with reproducible results.  Ultimately, we used hydra to test and reuse experiments in an easily iterable and reproducible manner.


## 7. Documentation & Repository Updates
- [X] **7.1 Updated README**
  - [X] Instructions for all new tools and processes
  -  [text](.hydra/config.yaml)
  -  [text](.hydra/hydra.yaml)

---

> **Checklist:** Use this as a guide for documenting your Phase 2 deliverables. Focus on operational robustness, reproducibility, and clear instructions for all tools and processes.
