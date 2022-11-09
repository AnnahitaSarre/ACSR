# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:06:34 2022

@author: hagar
"""

# Train Model Using Scikit Learn

# import relevant modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle # file to save the trained model
import os
import argparse
from utils import get_feature_names

parser = argparse.ArgumentParser()
parser.add_argument('--gender', default='male', choices=['male', 'female'])
parser.add_argument('--cropping', default='cropped', choices=['cropped', 'non_cropped'])
parser.add_argument('--path2features', default=os.path.join('..', 'output'))
parser.add_argument('--property-type', choices=['shape', 'position'],
                    default='shape')
parser.add_argument('--model-type', choices=['rf', 'lr', 'rc', 'gb'],
                    help = 'rf:random-forest; lr:logisitic-regrssion',
                    default='rf')
args = parser.parse_args()

df_features = pd.read_csv(os.path.join(args.path2features, f'training_features_{args.gender}_{args.cropping}.csv'),
                          index_col=0)

#run the pipline once for shape and once for pose
feature_names = get_feature_names(args.property_type)
# print('Training with the following features:')
# print(feature_names)
 
df_features = df_features[['fn_video'] + feature_names]# features
df_features = df_features.loc[df_features['fn_video'].str.contains(args.property_type,
                                                                   regex=False)]
y = df_features['fn_video'] # target value
X = df_features.drop(['fn_video'], axis=1)

# Split the data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1234)

pipelines = {'lr':make_pipeline(LogisticRegression()),
             'rc':make_pipeline(RidgeClassifier()),
             'rf':make_pipeline(RandomForestClassifier()),
             'gb':make_pipeline(GradientBoostingClassifier())}

pipeline = pipelines[args.model_type]
# fit the models with the pipelines
model = pipeline.fit(X_train, y_train)
# Evaluate and Serialize Model 
yhat = model.predict(X_test)
print(f'Performance for {args.property_type}, algrorithm {args.model_type}, is: {accuracy_score(y_test, yhat)}')


# Put the trained model in a pkl file\
os.makedirs(os.path.join('..', 'trained_models'), exist_ok=True)
file_name = os.path.join('..','trained_models',
                         f'model_{args.model_type}_{args.property_type}_{args.gender}_{args.cropping}.pkl')
with open(file_name, 'wb') as f:
    pickle.dump([model, feature_names], f)
