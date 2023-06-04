#!/bin/bash
set -x

model_type='rf' # 'rf', 'lr'
gender='female'
cropping='cropped'
path2test_videos='../stimuli/words/mp4'

counter=0
#for fn_video in $path2test_videos/word_*.mp4; do
#for fn_video in $path2test_videos/word_h2_13.mp4 $path2test_videos/word_h2_05.mp4; do
for fn_video in $path2test_videos/word_h2_05.mp4 $path2test_videos/word_h2_13.mp4 $path2test_videos/word_h2_22.mp4 $path2test_videos/word_h2_25.mp4 $path2test_videos/word_h2_30.mp4 $path2test_videos/word_h2_33.mp4 $path2test_videos/word_h2_36.mp4 $path2test_videos/word_h2_39.mp4 $path2test_videos/word_l0_07.mp4 $path2test_videos/word_l0_11.mp4 $path2test_videos/word_l0_18.mp4 $path2test_videos/word_l0_40.mp4 $path2test_videos/word_l1_11.mp4 $path2test_videos/word_l1_14.mp4 $path2test_videos/word_l1_19.mp4 $path2test_videos/word_l1_23.mp4 $path2test_videos/word_l1_26.mp4 $path2test_videos/word_l1_28.mp4 $path2test_videos/word_l1_30.mp4 $path2test_videos/word_l1_34.mp4 $path2test_videos/word_l2_01.mp4 $path2test_videos/word_l2_05.mp4 $path2test_videos/word_l2_10.mp4 $path2test_videos/word_l2_12.mp4 $path2test_videos/word_l2_13.mp4 $path2test_videos/word_l2_23.mp4 $path2test_videos/word_l2_27.mp4 $path2test_videos/word_l2_30.mp4 $path2test_videos/word_l2_31.mp4 $path2test_videos/word_l2_32.mp4 $path2test_videos/word_l2_34.mp4 $path2test_videos/word_l2_40.mp4; do
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
