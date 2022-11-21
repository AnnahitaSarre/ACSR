#!/bin/bash
# set -x

model_type='rf' # 'rf', 'lr'
gender='female' # male/female
cropping='cropped' # raw/cropped
path2test_videos='../stimuli/words/mp4'

echo python3 extract_training_coordinates.py --gender $gender --cropping $cropping #--show-video # !-SLOW-! For TRAINING VIDEOs ONLY
echo python3 extract_training_features.py --gender $gender --cropping $cropping
echo python3 train.py --property-type position --gender $gender --cropping $cropping --model-type $model_type
echo python3 train.py --property-type shape --gender $gender --cropping $cropping --model-type $model_type

counter=0
for fn_video in $path2test_videos/word_*.mp4; do
  let counter+=1
  fn_video="$(basename -- $fn_video)"
  echo '------------------------------------'
  echo $fn_video
  echo '------------------------------------'

  python3 predict.py --property-type position --gender $gender --cropping $cropping --path2test-videos $path2test_videos --fn-video $fn_video --model-type $model_type
  python3 predict.py --property-type shape --gender $gender --cropping $cropping --path2test-videos $path2test_videos --fn-video $fn_video --model-type $model_type
  python find_onsets.py --gender $gender --cropping $cropping --path2video $path2test_videos --fn-video $fn_video --model-type $model_type --textgrid
  python mark_video.py --gender $gender --cropping $cropping --path2video $path2test_videos --fn-video $fn_video --model-type $model_type --textgrid
done
echo $counter 'videos processed.'
