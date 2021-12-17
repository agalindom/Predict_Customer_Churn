# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This project is all about coding best practices in data science, here, a Churn prediction model is created in a modular way based on a notebook and written in a way that is ready to be production quality code.

All the scripts where edited based on the libraries `pylint` and `autopep8`.
*Note: all scripts have a pylint score >9*

## Files

**churn_library**: Module that contains all functions that runs and evaluates the model and the preprocessing steps.

**churn_script_logging_and_tests.py**: Moduel that contains tests to ensure that the code located in *churn_library* is running correctly.

**churn_notebook**: Notebook used to construct the model and from which the running script and test script are made.

**requirements.txt**: text file containing the neccessary libraries to correctly run the python script.

## Usage

Libraries can be installed via `pip` and all the necessary libraries can be installed at once by running the `requirements.txt` file in the shell:

> pip install -r requirements.txt

There are two ways that the model can be run.
The first one is via the `churn_script_logging_and_tests.py` script:

> python churn_script_logging_and_tests.py

- This will run a the model and also a series of tests to check that the script is running in the way it should be, the result of the test can be accessed in the `churn_library.log` file located at `./Data`.

The second way that the model can be run is via hte `churn_library.py` script

> python churn_library.py

- This will run the all the functions located at the script above but whithout
the testing.
