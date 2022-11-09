#!/bin/bash
set -x

model_type='rf' # 'rf', 'lr'
gender='female'
cropping='cropped'
path2test_videos='../stimuli/words/mp4'
path2test_videos='../data/test_videos'

#for fn_video in '001-1.mp4' '001-2.mp4' '002-1.mp4' '002-2.mp4'; do
# for fn_video in 'word_h0_01.mp4' 'word_h0_02.mp4'; do
for fn_video in '237-1.mp4'; do
	python3 predict.py --property-type position --gender $gender --cropping $cropping --path2test-videos $path2test_videos --fn-video $fn_video --model-type $model_type
        python3 predict.py --property-type shape --gender $gender --cropping $cropping --path2test-videos $path2test_videos --fn-video $fn_video --model-type $model_type
        python mark_video.py --gender $gender --cropping $cropping --path2video $path2test_videos --fn-video $fn_video --model-type $model_type --textgrid
done
