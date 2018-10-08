# Save Model Using Pickle
import pandas as pd
import os
from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
import pickle
import datetime
import numpy as np
import pickle

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


def print_model_evaluation_metrics(actual, predicted):
    print "Accuracy Score:\n%s\n" % metrics.accuracy_score(actual, predicted)
    print "Log Loss Score:\n%s\n" % metrics.log_loss(actual, predicted)
    print "Confusion Matrix:\n%s\n" % metrics.confusion_matrix(actual, predicted)
    percent_of_total_func = np.vectorize(lambda x: x/float(len(predicted)))
    print "Confusion Matrix (As Percents):\n%s\n" % percent_of_total_func(metrics.confusion_matrix(actual, predicted))
    print "Area Under the Receiver Operating Characteristic Curve:\n%s\n" % metrics.roc_auc_score(actual, predicted)
    print "F1 Score:\n%s\n" % metrics.f1_score(actual, predicted)
    print "Mean Absolute Error:\n%s\n" % metrics.mean_absolute_error(actual, predicted)
    print "Mean Squared Error:\n%s\n" % metrics.mean_squared_error(actual, predicted)
    return


def load_models(df, model_paths, generate_new=False):
    loaded_models = []

    if not generate_new:
        try:
            for model in model_paths:
                loaded_model  = pickle.load(open(model, 'rb'))
                loaded_models.append(loaded_model)
        except IOError as e:
            print e
            print "Please validate that a model exists at that file path."
    else:
        if len(model_paths) < 2:
            raise ValueError("You must inlcude at least two model file paths")
        generated_model_1 = generate_dummy_logistic_regression_model(df, file_name=model_paths[0], test_size=0.33, seed=7)
        generated_model_2 = generate_dummy_logistic_regression_model(df, file_name=model_paths[1], test_size=0.40, seed=6)
        loaded_models = [generated_model_1, generated_model_2]

    return loaded_models


# This expects:
#   - csv: to be a filepath to the input event stream data
#   - models: to be an array of filepaths .sav model files (and if they don't exist there, they will be created there)
#   - df_pickle_file: an optional argument that can be used to bypass the processing of the csv.
#   - generate_new_models: an option argument purely for internal use, used to generate two very similar models at the specified file paths.
def compare_models(csv, model_paths, df_pickle_file=None, generate_new_models=False):
    df = load_dataframe(df_pickle_file, csv=csv, overwrite_pickle=False)
    models = load_models(df, model_paths, generate_new=generate_new_models)

    array = df.values
    X = array[:,0:-1] # Input data to predict on
    Y = array[:,-1] # What actually happened

    for idx, model in enumerate(models):

        Y_predicted = predict_using_model(model, X)

        print "###########################################################"
        print "#########     Evaluation Metrics: Model %s      ############" % (idx+1)
        print "###########################################################\n"
        print_model_evaluation_metrics(Y, Y_predicted)
        print "\n"



if __name__ == "__main__":
    import sys
    from os import path
    path_to_project = path.dirname( path.dirname( path.abspath('__file__') ) )


    INPUT_CSV = './notebook/stored_csvs/TransactionsCompany1.csv'

    PICKLE_FILE_OF_DF = './notebook/stored_dataframes/dataframe.pkl'

    model_1_file = "./notebook/stored_models/model_1.sav"
    model_2_file = "./notebook/stored_models/model_2.sav"

    model_files = [model_1_file, model_2_file]

    compare_models(INPUT_CSV, model_files, df_pickle_file=PICKLE_FILE_OF_DF, generate_new_models=True)
