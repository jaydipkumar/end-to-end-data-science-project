from flask import Flask, flash, request, redirect, url_for, make_response, send_file, render_template
from werkzeug.utils import secure_filename
import io
import csv
import pickle
import os
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
app = Flask(__name__)
UPLOAD_FOLDER = '/uploads'

#model = pickle.load(open('model.pkl', 'rb'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/transform', methods=['POST'])
def transform_view():
    file = request.files['data_file']
    #file = request.files['data_file']
    if file:
        test = pd.read_csv(file, usecols=["ID", "ID_status", "active", "count_reassign", "count_opening",
       "count_updated", "ID_caller", "opened_by", "Created_by", "updated_by",
       "type_contact", "location", "category_ID", "user_symptom",
       "Support_group", "support_incharge", "Doc_knowledge",
       "confirmation_check", "notify"])
        """test = pd.DataFrame([file],columns=["ID", "ID_status", "active", "count_reassign", "count_opening",
       "count_updated", "ID_caller", "opened_by", "Created_by", "updated_by",
       "type_contact", "location", "category_ID", "user_symptom",
       "Support_group", "support_incharge", "Doc_knowledge",
       "confirmation_check", "notify"])"""
        train = pd.read_csv('train.csv')
        Y = train.impact
        train.drop(['impact'], axis=1, inplace=True)
        X = train
        del train
        X.drop(['Unnamed: 0'], axis=1, inplace=True)
        # Train Imputation
        X.drop(["opened_time", "created_at", "updated_at", "problem_ID", "change_request"], axis=1, inplace=True)
        #test.drop(['S.No', 'opened_time', 'created_at', 'updated_at', 'problem_ID', 'change_request'], axis=1, inplace=True)
        # Eliminate the NAN
        for col in X.columns:
            X.loc[X[col] == '?', col] = np.nan
        X.ID_caller = X.loc[X.ID_caller == '?', col] = 'Caller 1904'
        X.location = X.loc[X.location == '?', col] = 'Location 204'
        X.category_ID = X.loc[X.category_ID == '?', col] = 'Category 26'
        categorical = [col for col in X.columns if X[col].dtype == object]
        numerical = [col for col in X.columns if X[col].dtype != object]
        # instantiate both packages to use
        import sys
        sys.setrecursionlimit(100000)  # Increase the recursion limit of the OS
        for col in categorical:
            temp = {}
            count = 0
            for val in X[col].values:
                try:
                    temp[val]
                except:
                    temp[val] = count
                    count += 1
            for val in test[col].values:
                try:
                    temp[val]
                except:
                    temp[val] = count
                    count += 1
            X[col] = [temp[x] for x in X[col].values]
            test[col] = [temp[x] for x in test[col].values]
        Y = Y.to_frame()
        Impact_Predic_RF = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        Impact_Predic_RF.fit(X, Y)
        test_predict = Impact_Predic_RF.predict(test)
        test['impact'] = test_predict
        test.to_csv('impact_prediction.csv')
        #test_predict.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'result.csv'), index=False)
        #test_predict = "attachment; filename=result.csv"

        return send_file('impact_prediction.csv')

    else:
        return "No file"

    """stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    #print("file contents: ", file_contents)
    #print(type(file_contents))
    print(csv_input)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = transform(stream.read())"""


if __name__ == "__main__":
    app.run(debug=True)