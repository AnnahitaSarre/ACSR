# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:37:05 2022

"""
import pickle
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import textgrids
from sklearn.preprocessing import minmax_scale
from scipy.signal import argrelextrema


def load_model(filename):
    with open(filename, 'rb') as f:
        model, feature_names = pickle.load(f)
    return model, feature_names


def load_video(path2file):
    cap = cv2.VideoCapture(path2file)
    cap.set(3,640) # camera width
    cap.set(4,480) # camera height
    return cap


def extract_class_from_fn(fn):
    '''
    get class number from filename, e.g.,
    '4' from 'position_04.mp4'
    '''
    if fn is not None:
        st = fn.find('_') + 1
        ed = fn.find('.')
        return int(fn[st:ed])
    else:
        return None


def get_distance(df_name, landmark1, landmark2, norm_factor=None):
    '''


    Parameters
    ----------
    df_name : TYPE
        DESCRIPTION.
    landmark1 : STR
        name of first landmark (e.g., hand20)
    landmark2 : STR
        name of second landmark (e.g., face234)

    Returns
    -------
    series for dataframe
    The distance between landmark1 and landmark2

    '''

    x1 = df_name[f'x_{landmark1}']
    x2 = df_name[f'x_{landmark2}']
    y1 = df_name[f'y_{landmark1}']
    y2 = df_name[f'y_{landmark2}']
    z1 = df_name[f'z_{landmark1}']
    z2 = df_name[f'z_{landmark2}']
    d = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

    # NORMALIZE
    if norm_factor is not None:
        d /= norm_factor

    return  d

def get_delta_dim(df_name, landmark1, landmark2, dim, norm_factor=None):
    delta = df_name[f'{dim}_{landmark1}'] - df_name[f'{dim}_{landmark2}']
    # NORMALIZE
    if norm_factor is not None:
        delta /= norm_factor
    return  delta


def extract_coordinates(cap, fn_video, show_video=False, verbose=True):

    if verbose:
        print(f'Extracting coordinates for: {fn_video}')
    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions


    columns = ['fn_video', 'frame_number']
    num_coords_face = 468
    num_coords_hand = 21

    # generate columns names
    for val in range(0, num_coords_face):
        columns += ['x_face{}'.format(val), 'y_face{}'.format(val),
                      'z_face{}'.format(val), 'v_face{}'.format(val)]

    for val in range(0, num_coords_hand):
        columns += ['x_r_hand{}'.format(val), 'y_r_hand{}'.format(val),
                      'z_r_hand{}'.format(val), 'v_r_hand{}'.format(val)]

    df_coords = pd.DataFrame(columns=columns)

    n_frames = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
    if verbose:
        print(f'Number of frames in video: {n_frames}')
    pbar = tqdm(total=n_frames)

    # Initiate holistic model
    i_frame = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            i_frame += 1
            #print(f'{i_frame}/{n_frames}')
            if not ret:
                break
            # Recolor Feed

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)


            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            # 4. Pose Detections
            if show_video:
                # Draw face landmarks
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                         )

                # Right hand landmarks
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                         )
                # Pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                          )
                cv2.imshow('cued_estimated', image)


            # Export coordinates
            if results.face_landmarks is not None:
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z,
                                           landmark.visibility] for landmark in face]).flatten())

            else:
                face_row = [None] * 4
           # Extract right hand landmarks
            if results.right_hand_landmarks is not None:
                r_hand = results.right_hand_landmarks.landmark
                r_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z,
                                             landmark.visibility] for landmark in r_hand]).flatten())
            else:
                r_hand_row = [None] * 4


            #Create the row that will be written in the file
            row = [fn_video, i_frame] + face_row +r_hand_row
            curr_df = pd.DataFrame(dict(zip(columns, row)), index=[0])
            df_coords = pd.concat([df_coords, curr_df], ignore_index=True)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()

    assert df_coords.shape[0] == n_frames

    return df_coords



def extract_features(df_coords):
    #create the df of relevant feature

    df_features = pd.DataFrame()
    df_features['fn_video'] = df_coords['fn_video'].copy()
    df_features['frame_number'] = df_coords['frame_number']

    #face width to normalize the distance
    # print('Computing face width for normalization')
    face_width = get_distance(df_coords,'face234','face454').mean()
    norm_factor = face_width
    print(f'Face width computed for normalizaiton {face_width}')

    #norm_factor = None # REMOVE NORMALIZAION

    # HAND-FACE DISTANCES AS FEATURES FOR POSITION DECODING
    position_index_pairs = get_index_pairs('position')
    for hand_index, face_index in position_index_pairs:
        feature_name = f'distance_face{face_index}_r_hand{hand_index}'
        # print(f'Computing {feature_name}')
        df_features[feature_name] = get_distance(df_coords,
                                                  f'face{face_index}',
                                                  f'r_hand{hand_index}',
                                                  norm_factor=norm_factor)

        dx = get_delta_dim(df_coords,
                            f'face{face_index}',
                            f'r_hand{hand_index}',
                            'x',
                            norm_factor=norm_factor)

        dy = get_delta_dim(df_coords,
                            f'face{face_index}',
                            f'r_hand{hand_index}',
                            'y',
                            norm_factor=norm_factor)

        feature_name = f'tan_angle_face{face_index}_r_hand{hand_index}'
        df_features[feature_name] = dx/dy

    # HAND-HAND DISTANCES AS FEATURE FOR SHAPE DECODING
    shape_index_pairs = get_index_pairs('shape')
    for hand_index1, hand_index2 in shape_index_pairs:
        feature_name = f'distance_r_hand{hand_index1}_r_hand{hand_index2}'
        # print(f'Computing {feature_name}')
        df_features[feature_name] = get_distance(df_coords,
                                                 f'r_hand{hand_index1}',
                                                 f'r_hand{hand_index2}',
                                                 norm_factor=norm_factor)


    return df_features


def get_index_pairs(property_type):
    index_pairs = []
    if property_type == 'shape':
        index_pairs.extend([(2, 4), (5, 8), (9, 12), (13, 16), (17, 20),
                            (4, 5), (4, 8),
                            (8, 12), (7, 11), (6, 10)])

    elif property_type == 'position':
        hand_indices = [8, 9, 12] # index and middle fingers

        face_indices = [#0, # Middle Lips
                        #61, # right side of lips
                        #172, # right side down
                        #234, # right side up
                        130, # right corner of right eye
                        152, # chin
                        94 # nose
                        ]
        for hand_index in hand_indices:
            for face_index in face_indices:
                index_pairs.append((hand_index, face_index))

    return index_pairs


def get_feature_names(property_name):
    feature_names = []
    # POSITION
    if property_name == 'position':
        position_index_pairs = get_index_pairs('position')
        for hand_index, face_index in position_index_pairs:
            feature_name = f'distance_face{face_index}_r_hand{hand_index}'
            feature_names.append(feature_name)
            feature_name = f'tan_angle_face{face_index}_r_hand{hand_index}'
            feature_names.append(feature_name)
    # SHAPE
    elif property_name == 'shape':
        shape_index_pairs = get_index_pairs('shape')

        for hand_index1, hand_index2 in shape_index_pairs:
            feature_name = f'distance_r_hand{hand_index1}_r_hand{hand_index2}'
            feature_names.append(feature_name)

    return feature_names



def compute_predictions(model, df_features):
    '''
    model - sklean model
    df_features - dataframe with n_samples X n_features
    '''
    X = df_features.to_numpy()

    predicted_class, predicted_probs = [], []
    for X_i in X:
        if (None in X_i) or (np.nan in X_i) or any([xi!=xi for xi in X_i]):
            predicted_c = None
            predicted_p = None
        else:
            predicted_c = model.predict([X_i])[0]
            predicted_p = model.predict_proba([X_i])[0]
        predicted_class.append(predicted_c)
        predicted_probs.append(predicted_p)

    return np.asarray(predicted_probs, dtype=object), np.asarray(predicted_class)


def compute_velocity(df, landmark, fn=None):
    frame_number = df['frame_number']
    x = df['x_' + landmark].values
    y = df['y_' + landmark].values
    z = df['z_' + landmark].values

    dx = np.gradient(x, frame_number)
    dy = np.gradient(y, frame_number)
    dz = np.gradient(z, frame_number)

    dx2 = np.gradient(dx, frame_number)
    dy2 = np.gradient(dy, frame_number)
    dz2 = np.gradient(dz, frame_number)

    v = np.sqrt(dx**2 + dy**2 + dz**2)
    a = np.sqrt(dx2**2 + dy2**2 + dz2**2)

    v_smoothed = savgol_filter(v, 9, 3) # window
    a_smoothed = savgol_filter(a, 9, 3) # window

    if fn is not None:
        fig, ax = plt.subplots()
        ax.plot(v_smoothed, lw=3, color='k')
        ax.plot(a_smoothed, lw=3, color='b')
        ax.set_xlabel('Frame', fontsize=16)
        ax.set_ylabel('Velocity', fontsize=16)
        ax.set_ylim([-0.01, 0.01])
        fig.savefig(fn + '.png')
    return  v_smoothed, a_smoothed


def get_phone_onsets(fn_textgrid):
    times, labels = [], []

    grid = textgrids.TextGrid(fn_textgrid)
    phones = grid['phones']
    for phone in phones:
        if phone.text.transcode() != '':
            times.append(phone.xmin)
            labels.append(phone.text.transcode())

    return times, labels


def get_stimulus_string(fn_video):
    fn_base = os.path.basename(fn_video)[:-4]
    fn_stimulus = fn_base + '.txt'
    fn_stimulus = os.path.join('../stimuli/words/mfa_in', fn_stimulus)
    s = open(fn_stimulus, 'r').readlines()
    return s[0].strip('\n')


def dict_phone_transcription():
    # Megalex (key) to MFA (value) phone labels
    d = {}
    d['R'] = '??'
    d['N'] = '??'
    d['??'] = '????'
    d['Z'] = '??'
    d['5'] = '????'
    d['E'] = '??'
    d['9'] = '??'
    d['8'] = '??'
    d['S'] = '??'
    d['O'] = '??'
    d['2'] = '??'
    d['g'] = '??'
    d['g'] = '??'
    d['@'] = '????'
    d['8'] = '??'
    return d

def find_syllable_onsets(lpc_syllables, times_phones, labels_phones):
    phones = labels_phones.copy()
    d_phone_transcription = dict_phone_transcription()
    #print(lpc_syllables)
    #[print(p, t) for p, t in zip(phones, times_phones)]
    #print('-'*100)
    times = []
    for syllable in lpc_syllables:
        first_phone = syllable[0]
        if first_phone in d_phone_transcription.keys():
            first_phone = d_phone_transcription[first_phone]
        for i, phone in enumerate(phones):
            if first_phone == phone:
                times.append(times_phones[i])
                del phones[i]
                del times_phones[i]
                break
    return times


def get_syllable_onset_frames_from_lpc_file(fn_video):
    fn_base = os.path.basename(fn_video)[:-4]

    # Get LPC parsing of stimulus, into separate SYLLABLES
    # (MFA is for ALL phones and we need to know which phones are at the beginning of each syllable)
    fn_lpc_parsing = fn_base + '.lpc'
    fn_lpc_parsing = os.path.join('../stimuli/words/txt', fn_lpc_parsing)
    lpc_syllables = open(fn_lpc_parsing, 'r').readlines()[0].strip('\n').split()

    return lpc_syllables

    return 
def get_syllable_onset_frames_from_mfa(fn_video, lpc_syllables):

    # Load video and get number of frames per second (fps)
    cap = load_video(fn_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) # frames per second
    assert fps > 0; 'Frames per seconds is not a positive number'

    # Load corresponing TextGrid file
    fn_base = os.path.basename(fn_video)[:-4]
    fn_textgrid = fn_base + '.TextGrid'
    fn_textgrid = os.path.join('../stimuli/words/mfa_out', fn_textgrid)

    # Get LPC parsing of stimulus, into separate SYLLABLES
    # (MFA is for ALL phones and we need to know which phones are at the beginning of each syllable)
    #fn_lpc_parsing = fn_base + '.lpc'
    #fn_lpc_parsing = os.path.join('../stimuli/words/txt', fn_lpc_parsing)
    #lpc_syllables = open(fn_lpc_parsing, 'r').readlines()[0].strip('\n').split()

    # PHONE onests in seconds from MFA
    onset_secs_phones_mfa, labels_phones_textgrid = get_phone_onsets(fn_textgrid)
    print(onset_secs_phones_mfa, labels_phones_textgrid)
    # SYLLABLE ONSET from MFA based on the onset of their FIRST PHONE
    onset_secs_syllables_mfa = find_syllable_onsets(lpc_syllables, # in seconds
                                                    onset_secs_phones_mfa,
                                                    labels_phones_textgrid)
    onset_frames_syllables_mfa = [int(t*fps) for t in onset_secs_syllables_mfa] # in frames

    return onset_frames_syllables_mfa



def find_onsets_based_on_extrema(time_series,
                                 n_syllables=None,
                                 onset_frames_syllables_mfa=None,
                                 thresh=None): # condition: time_series > thresh

    if onset_frames_syllables_mfa is not None: 
        onset_frames_syllables_mfa = np.asarray(onset_frames_syllables_mfa)

    # find extrema
    onset_frames_extrema = argrelextrema(time_series, np.greater)[0]
    # Threshold
    if thresh is not None:
        onset_frames_extrema = np.asarray([onset_frame for onset_frame in onset_frames_extrema if time_series[onset_frame]>thresh])

    onset_frames_extrema_temp = onset_frames_extrema.copy()
    onset_frames_picked = []
    if onset_frames_syllables_mfa is not None: # use MFA onsets to constrain the solution
        if len(onset_frames_syllables_mfa) == len(onset_frames_extrema_temp):
            onset_frames_picked = onset_frames_extrema_temp
        else:
            for i_frame, onset_frame_syl_mfa in enumerate(onset_frames_syllables_mfa):
                # Find extremum that is nearest to current MFA onset
                delta = np.abs(onset_frames_extrema_temp - onset_frame_syl_mfa)
                IX_onset_frame_extremum_nearest_mfa = np.argmin(delta)
                onset_frame_extremum_nearest_mfa = onset_frames_extrema_temp[IX_onset_frame_extremum_nearest_mfa]
                onset_frames_picked.append(onset_frame_extremum_nearest_mfa)
                # Remove past indexes, in order to make sure the next onset frame is in the future
                onset_frames_extrema_temp = onset_frames_extrema_temp[onset_frames_extrema_temp > onset_frame_extremum_nearest_mfa]
                if len(onset_frames_extrema_temp)==0:
                    while len(onset_frames_picked) < len(onset_frames_syllables_mfa): # Fill None values if not enough identified extrema
                        onset_frames_picked.append(None)
                    break
    else:
        IXs = np.argpartition(onset_frames_extrema, -n_syllables)[-n_syllables:]
        onset_frames_picked = list(onset_frames_extrema[IXs])

    return onset_frames_picked, onset_frames_extrema

def scale_velocity(velocity):
    q25, q75 = np.percentile(velocity, 25), np.percentile(velocity, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    velocity = np.clip(velocity, lower, upper)
    velocity_scaled = minmax_scale(velocity)
    return velocity_scaled


def get_joint_measure(df_predictions_pos,
                      df_predictions_shape,
                      velocity_scaled,
                      weight_velocity=1):

    # MAX PROBABILITIES (POSITION AND SHAPE)
    max_probs_pos = df_predictions_pos.copy().filter(regex=("p_class*")).to_numpy().max(axis=1)
    max_probs_shape = df_predictions_shape.copy().filter(regex=("p_class*")).to_numpy().max(axis=1)
    probs_product = max_probs_pos * max_probs_shape
    # JOINT
    joint_measure = (weight_velocity * (1-velocity_scaled) + probs_product)/(1+weight_velocity)
    joint_measure_smoothed = savgol_filter(joint_measure, 15, 3) # window, smooth
    # replace nans caused by smoothing with original values
    is_nan_smoothed = np.isnan(joint_measure_smoothed)
    joint_measure_smoothed[is_nan_smoothed] = joint_measure[is_nan_smoothed]

    return joint_measure_smoothed


def write_onsets_to_file(str_stimulus, lpc_syllables, onset_frames_picked, fn_txt):
    
    # HACK TO EQUALIZE THE NUMBER OF EXPECTED ONSETS (NUM SYLLABLES) AND THE ONE FOUND
    #if len(lpc_syllables) < len(onset_frames_picked): # REMOVE EXTRA ONSETS
    #    onset_frames_picked = onset_frames_picked[:3]
    #for i_sy in range(len(lpc_syllables)-len(onset_frames_picked)): # ADD DUMMY ONSETS
    #    onset_frames_picked = list(onset_frames_picked)
    #    last_onset = onset_frames_picked[-1]
    #    onset_frames_picked.append(last_onset + i_sy + 1)

    assert len(lpc_syllables) == len(onset_frames_picked)

    with open(fn_txt, 'w') as f:
        f.write(f'{str_stimulus}\n')
        f.write('event,stimulus,frame_number\n')
        for (syllable, onset) in zip(lpc_syllables, onset_frames_picked):
            f.write(f'SYLLABLE ONSET, {syllable}, {onset}\n')
    return None
