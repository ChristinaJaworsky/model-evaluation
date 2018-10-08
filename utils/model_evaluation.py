# Save Model Using Pickle
import pandas as pd
import os
from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
import pickle
import datetime
import numpy as np
import pickle
import pprint as pp

from data_processing import load_dataframe


def generate_dummy_logistic_regression_model(df, file_name=None, test_size=0.33, seed=7, solver='lbfgs'):

    array = df.values
    X = array[:,0:-1]
    Y = array[:,-1]
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

    model = LogisticRegression(solver=solver)
    model.fit(X_train, Y_train)

    # Save our model
    if file_name:
        pickle.dump(model, open(file_name, 'wb'))

    return model


def predict_using_model(model, df_input):
    return model.predict(df_input)


def compute_model_evaluation_metrics(actual, predicted):

    percent_of_total_func = np.vectorize(lambda x: x/float(len(predicted)))

    model_evaluation_metrics = {
        'accuracy_score': {
            'label': 'Accuracy Score',
            'value': metrics.accuracy_score(actual, predicted)
        },
        'log_loss': {
            'label': 'Log Loss Score',
            'value': metrics.log_loss(actual, predicted)
        },
        'confusion_matrix': {
            'label': 'Confusion Matrix',
            'value': metrics.confusion_matrix(actual, predicted)
        },
        'confusion_matrix_percents': {
            'label': 'Confusion Matrix (As Percent of Whole)',
            'value': percent_of_total_func(metrics.confusion_matrix(actual, predicted))
        },
        'roc_auc_score': {
            'label': 'Area Under the Receiver Operating Characteristic Curve',
            'value': metrics.roc_auc_score(actual, predicted)
        },
        'f1_score': {
            'label': 'F1 Score',
            'value': metrics.f1_score(actual, predicted)
        },
        'mean_absolute_error': {
            'label': 'Mean Absolute Error',
            'value': metrics.mean_absolute_error(actual, predicted)
        },
        'mean_squared_error': {
            'label': 'Mean Squared Error',
            'value': metrics.mean_squared_error(actual, predicted)
        }
    }

    return model_evaluation_metrics

def print_model_evaluation_metrics(model_metrics_dict, model_number, model_name=None):

    print "###########################################################"
    print "#########     Evaluation Metrics: Model %s      ############" % (model_number or ' ')
    print "###########################################################\n"
    if model_name:
        print "Model Name: %s\n"
    print "%s:\n%s\n" % (model_metrics_dict['accuracy_score'].get('label'), model_metrics_dict['accuracy_score'].get('value'))
    print "%s:\n%s\n" % (model_metrics_dict['log_loss'].get('label'), model_metrics_dict['log_loss'].get('value'))
    print "%s:\n%s\n" % (model_metrics_dict['confusion_matrix'].get('label'), model_metrics_dict['confusion_matrix'].get('value'))
    print "%s:\n%s\n" % (model_metrics_dict['confusion_matrix_percents'].get('label'), model_metrics_dict['confusion_matrix_percents'].get('value'))
    print "%s:\n%s\n" % (model_metrics_dict['roc_auc_score'].get('label'), model_metrics_dict['roc_auc_score'].get('value'))
    print "%s:\n%s\n" % (model_metrics_dict['f1_score'].get('label'), model_metrics_dict['f1_score'].get('value'))
    print "%s:\n%s\n" % (model_metrics_dict['mean_absolute_error'].get('label'), model_metrics_dict['mean_absolute_error'].get('value'))
    print "%s:\n%s\n" % (model_metrics_dict['mean_squared_error'].get('label'), model_metrics_dict['mean_squared_error'].get('value'))
    print "%s:\n%s\n" % (model_metrics_dict['accuracy_score'].get('label'), model_metrics_dict['accuracy_score'].get('value'))
    print "\n"
    return


def load_models(df, model_paths, generate_new=False):
    loaded_models = {}

    if not generate_new:
        try:
            for model in model_paths:
                loaded_model  = pickle.load(open(model, 'rb'))
                loaded_models[model] = loaded_model
        except IOError as e:
            print e
            print "Please validate that a model exists at that file path."
    else:
        if len(model_paths) < 2:
            raise ValueError("You must inlcude at least two model file paths")
        generated_model_1 = generate_dummy_logistic_regression_model(df, file_name=model_paths[0], test_size=0.33, seed=7)
        loaded_models[model_paths[0]] = generated_model_1
        generated_model_2 = generate_dummy_logistic_regression_model(df, file_name=model_paths[1], test_size=0.40, seed=6)
        loaded_models[model_paths[1]] = generated_model_2

    return loaded_models


# This expects:
#   - csv: to be a filepath to the input event stream data
#   - models: to be an array of filepaths .sav model files (and if they don't exist there, they will be created there)
#   - df_pickle_file: an optional argument that can be used to bypass the processing of the csv.
#   - generate_new_models: an option argument purely for internal use, used to generate two very similar models at the specified file paths.
def compare_models(csv, model_paths, df_pickle_file=None, print_results=False, generate_new_models=False):
    df = load_dataframe(df_pickle_file, csv=csv, overwrite_pickle=False)
    models = load_models(df, model_paths, generate_new=generate_new_models)

    array = df.values
    X = array[:,0:-1] # Input data to predict on
    Y = array[:,-1] # What actually happened

    model_comparison = {}

    for idx, model_name in enumerate(models):
        model = models[model_name]
        Y_predicted = predict_using_model(model, X)
        model_evaluation_metrics = compute_model_evaluation_metrics(Y, Y_predicted)
        model_comparison[model_name] = model_evaluation_metrics

        if print_results:
            print_model_evaluation_metrics(model_evaluation_metrics, idx+1)

    return model_comparison



if __name__ == "__main__":
    import sys
    from os import path
    path_to_project = path.dirname( path.dirname( path.abspath('__file__') ) )

    INPUT_CSV = './notebook/stored_csvs/TransactionsCompany1.csv'
    PICKLE_FILE_OF_DF = './notebook/stored_dataframes/TransactionsCompany1.pkl'

    model_1_file = "./notebook/stored_models/model_1.sav"
    model_2_file = "./notebook/stored_models/model_2.sav"
    model_files = [model_1_file, model_2_file]

    model_comparison_results = compare_models(INPUT_CSV, model_files, df_pickle_file=PICKLE_FILE_OF_DF, print_results=False, generate_new_models=True)

    for idx, model_name in enumerate(model_comparison_results):
        model_results = model_comparison_results[model_name]
        print_model_evaluation_metrics(model_results, idx+1, model_name=model_name)
