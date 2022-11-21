#!/bin/bash
#set -x

model_type='rf' # 'rf', 'lr'
gender='female'
cropping='cropped'
path2test_videos='../stimuli/words/mp4'

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
~
~
~
