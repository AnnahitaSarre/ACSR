# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:23:06 2022

@author: hagar
"""

import os
import argparse
import pandas as pd
from utils import extract_features

# open csv with all the coords
parser = argparse.ArgumentParser()
parser.add_argument('--gender', default='male', choices=['male', 'female'])
parser.add_argument('--cropping', default='cropped', choices=['cropped', 'non_cropped'])
parser.add_argument('--path2coordinates', default=os.path.join('..', 'output'))
args=parser.parse_args()

# LOAD COORDINATE DATAFRAME
fn_coordinates = f'all_coords_face_hand_{args.gender}_{args.cropping}.csv'
df_coord = pd.read_csv(os.path.join(args.path2coordinates, fn_coordinates))

# CREATE NEW DATAFRAME FOR FEATURES
df_features = extract_features(df_coord)


# extract the df to a csv file
fn_features = f'training_features_{args.gender}_{args.cropping}.csv'
# print(df_features)
df_features.to_csv(os.path.join(args.path2coordinates, fn_features))
print(f'Data frame with features was saved to {os.path.join(args.path2coordinates, fn_features)}')
