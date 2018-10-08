# ML Model Evaluation Framework

This repo contains a framework for evaluating and comparing machine learning models. It utilizes a Jupyter notebook to walk-through the different components of this framework, their inputs, their outputs, and how they work.


## To Run Locally and Use the Notebook/Framework:
### Install Requirements & Start Jupyter:
- Open Terminal and cd into the repo. Create a virtual environment. Then, activate it and:
```
pip install -r requirements.txt
```
- Start Jupyter Notebook by running `jupyter notebook`. This should open a web browser tab with the directory.

### Uploading the Prerequisite Data:
- In the Jupyter browser tab, open `stored_csvs`. Upload any event-stream datasets that you would like to evaluate your model against. These csvs should be of the form `customer_id | event_timestamp | event_value` with one row per event.
- If you have your own models that you would like to evaluate, go back up a level, open the `stored_models` directory, and upload them here. They should be pickle dumps of a scikit-learn model with the .sav extension.

### Using the Notebook/Framework:
- Navigate back up to the `notebook` directory where you will find the file `Model Evaluation.ipynb`. Click to open.
- Read the directions in the Phase 0 section of the notebook to use the framework! You may also import the functions in the utils folder directly.
