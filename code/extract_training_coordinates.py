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

positions_list = [f'position_0{i}' for i in range(5)]
shapes_list = [f'shape_0{i}' for i in range(8)]

#positions_list = [f'position_{i}_from_words' for i in range(5)]
#shapes_list = [f'shape_{i}_from_words' for i in range(5)]

classes_list = [positions_list, shapes_list]

df = pd.DataFrame()

for fn_videos in classes_list:
    for fn_video in fn_videos:
        fn_video = os.path.join(args.path2data,
                                args.gender,
                                args.cropping,
                                fn_video+'.mp4')
        print(f'Loading: {fn_video}')
        cap = load_video(fn_video)
        # EXTRACT COORDINATES
        df_coords = extract_coordinates(cap, os.path.basename(fn_video),
                                        show_video=args.show_video,
                                        verbose=True)
        df = pd.concat([df,df_coords])

os.makedirs(args.path2output, exist_ok=True)

fn_output = f'training_coords_face_hand_{args.gender}_{args.cropping}.csv'    
fn_output = os.path.join(args.path2output, fn_output)
df.to_csv(fn_output)
print(f'coordinates saved to: {fn_output}')
