# Save Model Using Pickle
import pandas as pd
import os
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

csv = "~/Desktop/csvs/TransactionsCompany1.csv"
df = pd.read_csv(csv)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d')

# df['Timestamp'] = df['Timestamp'].astype(int)

array = df.values

print array


X = array[:,0:2]
Y = array[:,2]



test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)


# Fit the model on 33%
model = LogisticRegression()
# model.fit(X_train, Y_train)
# # save the model to disk
# path = os.path.abspath(os.path.dirname(__file__))
# filename = '%s/finalized_model.sav' % path
# pickle.dump(model, open(filename, 'wb'))
#
# # some time later...
#
# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)
