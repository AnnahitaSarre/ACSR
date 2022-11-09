#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import pandas as pd
from utils import extract_coordinates
from utils import load_video

parser = argparse.ArgumentParser()
parser.add_argument('--show-video', action='store_true', default=False)
parser.add_argument('--gender', default='female', choices=['male', 'female'])
parser.add_argument('--cropping', default='cropped', choices=['cropped', 'non_cropped'])
parser.add_argument('--path2data', default=os.path.join('..', 'data',
                                                        'training_videos'))
parser.add_argument('--path2output', default=os.path.join('..', 'output'))
args = parser.parse_args()

file_name = f'all_coords_face_hand_{args.gender}_{args.cropping}.csv'    

positions_list = ['position_00','position_01','position_02',
                  'position_03','position_04'] 

shapes_list = ['shape_00','shape_01','shape_02','shape_03',
               'shape_04','shape_05','shape_06','shape_07']

classes_list = [positions_list, shapes_list]

df = pd.DataFrame()

for fn_videos in classes_list:
    for fn_video in fn_videos:
        fn_video = os.path.join(args.path2data,
                                args.gender,
                                args.cropping,
                                fn_video+'.mp4')
        print(f'Extracting coordinates from: {fn_video}')
        cap = load_video(fn_video)
        df_coords = extract_coordinates(cap, os.path.basename(fn_video),
                                        show_video=args.show_video)
        df = pd.concat([df,df_coords])

df.to_csv(os.path.join(args.path2output, file_name))
