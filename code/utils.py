# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:37:05 2022

@author: hagar
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
    st = fn.find('_') + 1
    ed = fn.find('.')
    return int(fn[st:ed])


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
    
    
def extract_coordinates(cap, fn_video, show_video=False):
    
    
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
            try:
                # Extract Face landmarks
                
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z,
                                           landmark.visibility] for landmark in face]).flatten())
               
               # Extract right hand landmarks
                r_hand = results.right_hand_landmarks.landmark
                r_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z,
                                             landmark.visibility] for landmark in r_hand]).flatten())

              
                
                #Create the row that will be written in the file
                row = [fn_video, i_frame] + face_row +r_hand_row
                df_coords = df_coords.append(dict(zip(columns, row)),
                                             ignore_index=True)

            except:
                pass


            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            pbar.update(1)
            
    cap.release()
    cv2.destroyAllWindows()
    
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
    predicted_class = model.predict(X)
    predicted_probs = model.predict_proba(X)

    return predicted_probs, np.asarray(predicted_class)
 
 
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


def get_stimulus_string(fn_stimulus):
    s = open(fn_stimulus, 'r').readlines()
    return s[0].strip('\n') 
