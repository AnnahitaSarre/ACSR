# Automatic Cued-Speech Recognition (ACSR)
Automatic detection of hand position and shape for cued-speech recognition. The ACSR is based on MediaPipe, which is used to extract skeleton landmarks for hand and face. The decoders are trained on features computed from the landmarks, and the results are then marked on the video and saved as a separate file. 

## Pipeline
To train an ACSR and test it on your video, follow the steps below:

![ACSR](https://github.com/HagarSalpeter/Decoder/blob/master/data/test_videos/test_marked.png)

[An example of ACSR on a test video](data/test_videos/test_marked.avi)

1. Extract skeleton coordinates (hand and face landmarks - see cartoons below) from the traing videos, which are in the folder: `data/training_videos`. Results are then saved as a csv file to `output/`:

   `python extract_training_coordinates.py --gender female --cropping cropped --show-video`

2. Compute various features based on the extracted skeleton coordinates from the previous step (hand-nose distance, finger lengths, etc.). Results are then saved as a csv file to `output/`:

   `python extract_training_features.py --gender female --cropping cropped`

3. Train two separate random-forest models for hand position and shape detection. The trained models are saved to `trained_models/`:

   `python train.py --property-type position --gender female --cropping cropped --model-type rf`
   `python train.py --property-type shape --gender female --cropping cropped --model-type rf`

4. Generate predictions for hand position and shape for a new test video. Similarlty to the train videos, the test video first goes through coordinate extraction and feature computations. Once the features are computed, predictions are made based on the trained models for hand position and shape. Predictions are saved as two separate csv files to `output/`:

   `python predict.py --property-type position --gender female --cropping cropped --path2test-videos ../stimuli/words/mp4 --fn-video word_h0_01.mp4 --model-type rf`
   `python3 predict.py --property-type shape --gender female --cropping cropped --path2test-videos ../stimuli/words/mp4 --fn-video word_h0_01.mp4 --model-type rf`

5. Compute hand velocity and other measures, from which event onsets are extracted, and saved to an event file. (e.g., `data/test_videos/word_h0_02.mp4.events`).
   `python find_onsets.py --gender female --cropping cropped --path2video ../stimuli/words/mp4 --fn-video word_h0_01.mp4 --model-type rf --textgrid`

6. Mark the predictions for hand position and shape on the test video, together with the marking of the landmarks and the event onsets. The marked video is saved at the same path as the test video (e.g., `data/test_videos/`) with an additional `_marked` ending (e.g., `test_marked.mp4`):

   `python mark_video.py --gender female --cropping cropped --path2video ../stimuli/words/mp4 --fn-video word_h0_01_pseudo.mp4 --model-type rf --textgrid`

### Landmark pose map: ###
![alt text](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)

### Landmark hand map: ###
![alt text](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)

### Landmark face map: ###
![alt text](https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)
