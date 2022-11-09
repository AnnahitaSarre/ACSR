# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:08:44 2022

@author: hagar
"""
import argparse
import os
import pandas as pd

from utils import load_model, load_video, compute_velocity
from viz import mark_pred_on_video

parser = argparse.ArgumentParser()
parser.add_argument('--gender', default='male', choices=['male', 'female'])
parser.add_argument('--cropping', default='cropped', choices=['cropped', 'non_cropped'])
parser.add_argument('--model-type', choices=['rf', 'lr', 'rc', 'gb'],
                    help = 'rf:random-forest; lr:logisitic-regrssion',
                    default='rf')
parser.add_argument('--fn-video', default='word_h0_02.mp4')
parser.add_argument('--path2video', default=os.path.join('..', 'data',
                                                         'test_videos'))
parser.add_argument('--path2predictions', default=os.path.join('..',
                                                               'output'))
parser.add_argument('--path2output', default=os.path.join('..', 'output'))
parser.add_argument('--text-factor', default=1, type=float)
parser.add_argument('--textgrid', action='store_true', default=False,
                    help='If true, onset from grid text will be added')
parser.add_argument('--show-video', action='store_true', default=False)
args = parser.parse_args()

# LOAD VIDEO
fn_video = os.path.join(args.path2video, args.fn_video)
cap = load_video(fn_video)
print(f'Visualization for: {fn_video}')
print(cap.__sizeof__())

# LOAD PREDICTIONS
fn_predictions_pos = f'predictions_{args.model_type}_position_{args.gender}_{args.cropping}_{args.fn_video[:-4]}.csv'
df_predictions_pos = pd.read_csv(os.path.join(args.path2predictions, fn_predictions_pos))
fn_predictions_shape = f'predictions_{args.model_type}_shape_{args.gender}_{args.cropping}_{args.fn_video[:-4]}.csv'
df_predictions_shape = pd.read_csv(os.path.join(args.path2predictions, fn_predictions_shape))

# LOAD COORDINATE DATAFRAME
df_coord = pd.read_csv(os.path.join(args.path2output,
                                    f'{args.fn_video[:-4]}_coordinates.csv'))

# LOAD FEATURES
df_features = pd.read_csv(os.path.join(args.path2output,
                                       f'{args.fn_video[:-4]}_features.csv'))

velocity, acceleration = compute_velocity(df_coord, 'r_hand9', 
                                          fn=f'../output/velocity_{args.fn_video}')
   

# print(velocity.shape)

# print(df_predictions_pos, df_predictions_shape)
fig_joint, ax_join = mark_pred_on_video(cap, fn_video, args.gender, args.cropping,
                                        df_predictions_pos, df_predictions_shape,
                                        velocity, acceleration, 
                                        velocity_thresh=0.008,
                                        acceleration_thresh=0.003,
                                        p_thresh=0.5,
                                        text_factor=args.text_factor,
                                        textgrid=args.textgrid,
                                        show=args.show_video)

print(f'The marked video was saved to: {fn_video}')
fig_joint.savefig(f'../figures/{fn_video}.png')
print(f'Figure was save to: ../figures/{fn_video}.png')
