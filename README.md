# Automatic Cued-Speech Recognition (ACSR)
Automatic detection of hand position and shape for cued-speech recognition. The ACSR is based on MediaPipe, which is used to extract skeleton landmarks for hand and face. The decoders are trained on features computed from the landmarks, and the results are then marked on the video and saved as a separate file. 

## Pipeline
To train an ACSR and test it on your video, follow the steps below:

![ACSR](https://github.com/HagarSalpeter/Decoder/blob/master/data/test_videos/test_marked.png)

[An example of ACSR on a test video](data/test_videos/test_marked.avi)

1. This step extract skeleton coordinates (hand and face landmarks - see cartoons below) from the traing videos, which are in the folder: `data/training_videos`. Results are then saved as a csv file to `output/`:

   `extract_coordinates.py --show-video`

2. This step computes various features based on the extracted skeleton coordinates from the previous step (hand-nose distance, finger lengths, etc.). Results are then saved as a csv file to `output/`:

   `extract_features.py`

3. This step trains two separate random-forest models for hand position and shape detection. The trained models are saved to `trained_models/`:

   `train.py --property-type position --model-type rf`
   `train.py --property-type shape --model-type rf`

4. This step generates predictions for hand position and shape for a new test video. Similarlty to the train videos, the test video first goes through coordinate extraction and feature computations. Once the features are computed, predictions are made based on the trained models for hand position and shape. Predictions are saved as two separate csv files to `output/`:

   `predict.py --property-type position --model-type rf --fn-video test.mp4`
   `predict.py --property-type shape --model-type rf --fn-video test.mp4`

5. This step marks the predictions for hand position and shape on the test video, together with the marking of the landmarks. The marked video is saved at the same path as the test video (e.g., `data/test_videos/`) with an additional `_marked` ending (e.g., `test_marked.mp4`):

   `mark_video.py --model-type rf --fn-video test.mp4`

### Landmark pose map: ###
![alt text](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)

### Landmark hand map: ###
![alt text](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)

### Landmark face map: ###
![alt text](https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)
