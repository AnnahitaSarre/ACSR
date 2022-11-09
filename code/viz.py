#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:20:23 2022

@author: yl254115
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
from utils import get_phone_onsets
from utils import get_stimulus_string
from PIL import ImageFont, ImageDraw, Image
from scipy.signal import argrelextrema
from sklearn.preprocessing import minmax_scale
from scipy.signal import savgol_filter

def open_cartoon(fn_cartoon):
    cartoon = cv2.imread(fn_cartoon, cv2.IMREAD_COLOR)
    return cartoon


def find_syllable_onsets(lpc_syllables, times_phones, labels_phones):
    phones = labels_phones.copy()
    #print(lpc_syllables)
    #[print(p, t) for p, t in zip(phones, times_phones)]
    #print('-'*100)
    times = []
    c=0
    for syllable in lpc_syllables:
        first_phone = syllable[0]
        for i, phone in enumerate(phones):
            if first_phone == phone:
                t = times_phones[i+c]
                times.append(t)
                #print(syllable, first_phone, phone, t)
                del phones[i]
                c += 1
                break


    return times

def find_minimal_velocity(joint_info, frames_syllables, index2frame, with_textgrid):
    n_syllables = len(frames_syllables)
    maxima = argrelextrema(joint_info, np.greater)[0]
    maxima = np.asarray([i_frame for i_frame in maxima if joint_info[i_frame]>0.3])
    i_frame_minima = maxima.copy()
    frames_syllables = np.asarray(frames_syllables)
    print('SYLLABLE ONSETS (IN FRAMES):')
    print(frames_syllables)
    i_frame_minima = np.asarray([index2frame[i] for i  in i_frame_minima])
    print('MAXIMA (IN FRAMES)')
    print(i_frame_minima)

    onsets = []
    if with_textgrid:
        for i_frame, frame_syl in enumerate(frames_syllables):
            delta = np.abs(i_frame_minima - frame_syl)
            i_frame_min = np.argmin(delta)
            onset = i_frame_minima[i_frame_min]
            i_frame_minima = i_frame_minima[i_frame_minima>onset]
            onsets.append(onset)
    else:
        IXs = np.argpartition(maxima, -n_syllables)[-n_syllables:]
        onsets = list(maxima[IXs])
    return onsets, maxima


def mark_pred_on_video(cap, fn_video, gender, cropping,
                       df_predictions_pos, df_predictions_shape,
                       velocity, acceleration,
                       velocity_thresh=0.01,
                       acceleration_thresh=0.003,
                       p_thresh=0.5,
                       text_factor=1,
                       textgrid=False,
                       plot_joint=True,
                       show=False):
   
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) # frames per second
    pbar = tqdm(total=n_frames)
    
    index2frame = dict(zip(df_predictions_pos.index.to_list(), df_predictions_pos['frame_number'].to_list()))
    
    # MAX PROBABILITIES (POSITION AND SHAPE)
    max_probs_pos = df_predictions_pos.copy().filter(regex=("p_class*")).to_numpy().max(axis=1)
    max_probs_shape = df_predictions_shape.copy().filter(regex=("p_class*")).to_numpy().max(axis=1)
    probs_product = max_probs_pos * max_probs_shape
    # SCALE VELOCITY
    q25, q75 = np.percentile(velocity, 25), np.percentile(velocity, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    velocity = np.clip(velocity, lower, upper)
    velocity_scaled = minmax_scale(velocity)
    # JOINT
    joint_info = (1-velocity_scaled) * probs_product
    #joint_info = (1-velocity_scaled)
    joint_info = savgol_filter(joint_info, 15, 3) # window

    fn_base = os.path.basename(fn_video)[:-4] 
    fn_textgrid = fn_base + '.TextGrid'
    fn_textgrid = os.path.join('../stimuli/words/mfa_out', fn_textgrid)
    times_phones, labels_phones = get_phone_onsets(fn_textgrid)
    frames_phones = [int(t*fps) for t in times_phones]

    fn_stimulus = fn_base + '.txt'
    fn_stimulus = os.path.join('../stimuli/words/mfa_in', fn_stimulus)
    str_stimulus = get_stimulus_string(fn_stimulus)

    fn_lpc_parsing = fn_base + '.lpc'
    fn_lpc_parsing = os.path.join('../stimuli/words/txt', fn_lpc_parsing)
    lpc_syllables = open(fn_lpc_parsing, 'r').readlines()[0].strip('\n').split()
    times_syllables = find_syllable_onsets(lpc_syllables,
                                           times_phones, labels_phones)
    frames_syllables = [int(t*fps) for t in times_syllables] 
    
    if textgrid:
        onsets, i_frame_minima = find_minimal_velocity(joint_info, frames_syllables,
                                                       index2frame, True) 
    else:
        onsets, i_frame_minima = find_minimal_velocity(joint_info, frames_syllables,
                                                       index2frame, False) 

    print('Syllable, frame number (mfa), frame number (onset)')
    [print(syl, frame_mfa, onset) for syl,frame_mfa,onset in zip(lpc_syllables, frames_syllables, onsets)]
    assert len(lpc_syllables) == len(times_syllables) == len(frames_syllables) == len(onsets)
    

    fig, ax = None, None
    if plot_joint:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(df_predictions_pos['frame_number'].to_list(), joint_info, 'k', lw=3)
        ax.set_xlabel('Frame', fontsize=14)
        ax.set_ylabel('max(prob_pos)*max(prob_shape)*(1-minmax(velocity))', fontsize=14)
        ax.plot(df_predictions_pos['frame_number'].to_list(), velocity_scaled, 'r', lw=1)
        ax.plot(df_predictions_pos['frame_number'].to_list(), max_probs_pos, 'g', lw=1)
        ax.plot(df_predictions_pos['frame_number'].to_list(), max_probs_shape, 'b', lw=1)
        ax.set_xticks(frames_syllables)
        ax.set_xticklabels(lpc_syllables)
        
        for onset in onsets:
            ax.axvline(onset, color='k', ls='--')

    #return fig, ax

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions
    
    size = (int(cap.get(3)), int(cap.get(4)))
    marked_video = cv2.VideoWriter(f'{fn_video[:-4]}_marked_with_model_{gender}_{cropping}.avi',
                                   cv2.VideoWriter_fourcc(*'XVID'),30,
                                   size)
    


    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        i_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            i_frame += 1
            if not ret:
                print('!'*100)
                break
            
            # POSITION
            curr_row_pos = df_predictions_pos[df_predictions_pos['frame_number']==i_frame]
            # SHAPE
            curr_row_shape = df_predictions_shape[df_predictions_shape['frame_number']==i_frame]
            if curr_row_shape.empty or curr_row_pos.empty: # Process only 
                pbar.update(1)
                continue
            
            
            # GET PREDICTIONS
            i_df = curr_row_shape.index
            predicted_class_pos = curr_row_pos['predicted_class'].values[0]
            predicted_probs_pos = curr_row_pos[f'p_class_{predicted_class_pos + 1}'].values[0]
            
            predicted_class_shape = curr_row_shape['predicted_class'].values[0]
            predicted_probs_shape = curr_row_shape[f'p_class_{predicted_class_shape + 1}'].values[0]
            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                     )
            
            # Right hand landmarks
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
            
       
            # Open Cartoon        
            fn_cartoon = f'pos_n{predicted_class_pos}_shape_n{predicted_class_shape}.png' 
            fn_cartoon = os.path.join('../data/cartoons/', fn_cartoon)
            cartoon = open_cartoon(fn_cartoon)
            height, width, channels = cartoon.shape
            
            # Write prediction on video:
            font = cv2.FONT_HERSHEY_SIMPLEX
            line_type = cv2.LINE_8 #LINE_AA            
            
            x1_box = int(50/text_factor)
            x2_box = int(100/text_factor)
            y1_box = int(80/text_factor)
            y2_box = y1_box+height
            
            x_cartoon = int(80/text_factor)
            y_cartoon = int(100/text_factor)

            x1_text = int(x_cartoon/text_factor)
            x2_text = int((x_cartoon+300)/text_factor)
            dy_text = int(50/text_factor)
            y_text = int(800/text_factor)
            
            offset = np.array((x_cartoon, y_cartoon)) #top-left point from which to insert the smallest image. height first, from the top of the window
            image[offset[0]:offset[0] + height,
            offset[1]:offset[1] + width] = cartoon

            
            cv2.putText(image, 'SHAPE:', (x1_text, y_text),
                        font, 1/text_factor, (255, 255, 255), 2, line_type)
            for i_shape in range(1, 9):
                y_text += dy_text
                cv2.putText(image, f'p_class_{i_shape}', (x1_text, y_text),
                            font, 1/text_factor, (255, 255, 255), 2, line_type)
                p_class = curr_row_shape[f'p_class_{i_shape}'].values[0]
                cv2.putText(image, str(p_class), (x2_text, y_text),
                            font, 1/text_factor, (255, 255, 255), 2, line_type)
            
            y_text += dy_text
            cv2.putText(image, 'POSITION:', (x1_text, y_text),
                        font, 1/text_factor, (255, 255, 255), 2, line_type)
            for i_pos in range(1, 6):
                y_text += dy_text
                cv2.putText(image, f'p_class_{i_pos}', (x1_text, y_text),
                            font, 1/text_factor, (255, 255, 255), 2, line_type)
                p_class = curr_row_pos[f'p_class_{i_pos}'].values[0]
                cv2.putText(image, str(p_class), (x2_text, y_text),
                            font, 1/text_factor, (255, 255, 255), 2, line_type)
            
            y_text += dy_text
            cv2.putText(image, 'Velocity', (x1_text, y_text),
                        font, 1/text_factor, (255, 255, 255), 2, line_type)
            cv2.putText(image, f'{velocity[i_df][0]:1.5f}', (x2_text, y_text),
                        font, 1/text_factor, (255, 255, 255), 2, line_type)
            if i_frame in i_frame_minima:
                cv2.putText(image, '*', (int(x1_text/2), y_text),
                            font, 1/text_factor, (255, 255, 255), 2, line_type)
            
            y_text += dy_text
            cv2.putText(image, 'Frame', (x1_text, y_text),
                        font, 1/text_factor, (255, 255, 255), 2, line_type)
            cv2.putText(image, f'{i_frame}', (x2_text, y_text),
                        font, 1/text_factor, (255, 255, 255), 2, line_type)
       
            # mark sentence
            font = ImageFont.truetype("DejaVuSans.ttf", 32)
            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)
            x_stim = 10 
            y_stim = int(y_cartoon/3)
            draw.text((x_stim, y_stim), str_stimulus, font=font)
            image = np.array(img_pil)
            
            # Mark forced alignment
            x_phone = int(x_cartoon + width*1.2)
            y_phone = y_cartoon
            if i_frame in frames_phones:
                IX = frames_phones.index(i_frame)
                label_phone = labels_phones[IX]
                
                font = ImageFont.truetype("DejaVuSans.ttf", 32)
                img_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x_phone, y_phone), label_phone, font=font)
                image = np.array(img_pil)
            
            
            # Mark forced alignment
            x_onset = int(x_cartoon + width*0.4)
            y_onset = int(y_cartoon+height*1.2)
            if i_frame in onsets:
                IX = onsets.index(i_frame)
                syl = lpc_syllables[IX]
                
                font = ImageFont.truetype("DejaVuSans.ttf", 32)
                img_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x_onset, y_onset), f'ONSET ({syl})', font=font)
                image = np.array(img_pil)

            # Mark prediction
            if predicted_probs_pos > p_thresh and \
                predicted_probs_shape > p_thresh and \
                    velocity[i_df][0]<velocity_thresh:# and \
                        
                        cv2.rectangle(image, (x1_box, y1_box), (x2_box, y2_box),
                                      (16, 255, 16), -1)
             

                    
            # Mark onsets
            #if np.any([i_frame > frame_phone - fps/5 and i_frame < frame_phone + fps/5 for frame_phone in frames_phones]):
            #    is_around_phone_onsets = True
            #else:
            #    is_around_phone_onsets = False

            #if predicted_probs_pos > 0.35 and \
            #    predicted_probs_shape > 0.35 and \
            #        velocity[i_df][0]<0.1 and \
            #        is_around_phone_onsets:
            #            font = cv2.FONT_HERSHEY_SIMPLEX
            #            cv2.putText(image, 'ONSET', (x_phone, y_phone*2),
            #                        font, 2/text_factor, (255, 1, 1), 2, line_type)

    
            if show:
                cv2.imshow('cued_estimated', image)
            # print(image)
            marked_video.write(image)
    
    
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            pbar.update(1)
    
    marked_video.release()
    cap.release()
    cv2.destroyAllWindows()
    print("The video was successfully saved")
    return fig, ax

def plot_predictions(df_predictions_pos, df_predictions_shape, velocity,
                     thresh=0.5):
    fig, ax = plt.subplots(figsize=(15, 10))
    
    df_predictions_pos = df_predictions_pos.filter(regex=("p_class*"))
    df_predictions_shape = df_predictions_shape.filter(regex=("p_class*"))
    
    probs_pos = df_predictions_pos.to_numpy()
    # probs_pos[probs_pos<thresh] = np.nan
    #ax.plot(probs_pos, ls='-', lw=2,
    #        label=['pos_' + s for s in df_predictions_pos.columns])
    
    probs_shape = df_predictions_shape.to_numpy()
    # probs_shape[probs_shape<thresh] = np.nan
    #ax.plot(probs_shape, ls='--', lw=2,
    #        label=['shape_' + s for s in df_predictions_shape.columns])
    
    ax.set_xlabel('Frame number', fontsize=16)
    ax.set_ylabel('Probability', fontsize=16)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # ax.set_ylim((0.7, 1.1))
    
    probs_product = np.max(probs_pos, axis=1)*np.max(probs_shape, axis=1)
    ax.plot(probs_product, color='k', lw=1)
    
    ax.plot(1-minmax_scale(velocity), 'r', lw=1)
    ax.plot((1-minmax_scale(velocity))*probs_product, 'g', lw=4)

    plt.subplots_adjust(right=0.8)
    
    return fig, ax


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
