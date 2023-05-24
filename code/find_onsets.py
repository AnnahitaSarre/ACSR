# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:08:44 2022

"""
import argparse
import os
import pandas as pd
from pathlib import Path
import PyQt5
import utils
import viz

parser = argparse.ArgumentParser()
parser.add_argument('--gender', default='female', choices=['male', 'female'])
parser.add_argument('--cropping', default='cropped', choices=['cropped', 'non_cropped'])
parser.add_argument('--model-type', choices=['rf', 'lr', 'rc', 'gb'],
                    help = 'rf:random-forest; lr:logisitic-regrssion',
                    default='rf')
parser.add_argument('--fn-video', default='word_h0_11.mp4')
parser.add_argument('--path2video', default=os.path.join('..', 'stimuli',
                                                         'words', 'mp4'))
parser.add_argument('--path2predictions', default=os.path.join('..',
                                                               'output'))
parser.add_argument('--path2output', default=os.path.join('..', 'output'))
parser.add_argument('--path2figures', default=os.path.join('..', 'figures'))
#parser.add_argument('--n-syllables', default=None,
#                    help='If textgrid is true then num syllables taken from MFA')
parser.add_argument('--textgrid', action='store_true', default=False,
                    help='If true, onset from grid text will be added')
parser.add_argument('--plot-measures', action='store_true', default=True,
                    help='If true, velocity, joint measures and probabilites will be plotted')
parser.add_argument('--weight-velocity', default=3,
                    help='Importance weight of velocity compared to probabilities in the computation of the joint measure')
args = parser.parse_args()

#if args.textgrid:
#    args.n_syllables = None
#else:
#    if args.n_syllables is None:
#        raise('If textgrid is False then n-syllables must be provided!')

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
)


# LOAD VIDEO
fn_video = os.path.join(args.path2video, args.fn_video)
cap = utils.load_video(fn_video)
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

# COMPUTE VELOCITY AND SCALE IT WITH MINMAX
velocity, acceleration = utils.compute_velocity(df_coord, 'r_hand9', 
                                                fn=f'../output/velocity_{args.fn_video}')
   
velocity_scaled = utils.scale_velocity(velocity)

# GET STIMULUS ENTIRE STRING
str_stimulus = utils.get_stimulus_string(fn_video)

# GET SYLLABLE ONSETS FROM MFA
lpc_syllables = utils.get_syllable_onset_frames_from_lpc_file(fn_video)
n_syllables = len(lpc_syllables)
print(lpc_syllables)

if args.textgrid:
    onset_frames_syllables_mfa = utils.get_syllable_onset_frames_from_mfa(fn_video, lpc_syllables)
    print(f'Number of MFA syllabels found: {n_syllables}')
else:
    lpc_syllables = None
    onset_frames_syllables_mfa = None

joint_measure = utils.get_joint_measure(df_predictions_pos,
                                        df_predictions_shape,
                                        velocity_scaled,
                                        weight_velocity=args.weight_velocity)

print(f'Stimulus: {str_stimulus}')
if (lpc_syllables is not None) and (onset_frames_syllables_mfa is not None):
    [print(f'{syl}: frame #{onset}') for syl, onset in zip(lpc_syllables, onset_frames_syllables_mfa)]


# GET ONSET FRAMES BASED ON JOINT MEASURE (PROBABILITIES, VELOCITY AND POSSIBLY MFA)
thresh = 0.3
onset_frames_picked, onset_frames_extrema = utils.find_onsets_based_on_extrema(joint_measure,
                                                                               n_syllables,
                                                                               onset_frames_syllables_mfa=onset_frames_syllables_mfa,
                                                                               thresh=thresh)
print('Frame onsets of extrema of joint measure:', onset_frames_extrema)
print('Identified frame onsets:', onset_frames_picked)

# PLOT JOINT MEASURE, SCALED VELOCITY AND PROBS (POSITION AND SHAPE)
if args.plot_measures:
    fig, _ =  viz.plot_joint_measure(df_predictions_pos,
                                     df_predictions_shape,
                                     velocity_scaled,
                                     joint_measure,
                                     lpc_syllables,
                                     onset_frames_syllables_mfa,
                                     onsets_extrema=onset_frames_picked) # without extrema onsets

    os.makedirs(args.path2figures, exist_ok=True)
    fn_fig = os.path.join(args.path2video, f'{os.path.basename(fn_video)}.png')
    fig.savefig(fn_fig)
    print(f'Figure was save to: {fn_fig}')


os.makedirs(args.path2output, exist_ok=True)
fn_txt = os.path.join(args.path2video, f'{os.path.basename(fn_video)}.events')
utils.write_onsets_to_file(str_stimulus, lpc_syllables, onset_frames_picked, fn_txt)
print(f'Event onsets saved to: {fn_txt}')

# SAVE MESAURES TO CSV
df_measures = pd.DataFrame(list(zip(velocity, acceleration, velocity_scaled, joint_measure)),
                           columns=['velocity', 'acceleration', 'velocity_scaled', 'joint_measure'])
df_measures.to_csv(os.path.join(args.path2output,
                                f'{args.fn_video[:-4]}_measures.csv'))

