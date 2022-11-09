#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/network/lustre/iss02/cohen/data/Annahita/LPC/ACSR/Decoder/code/
#SBATCH --output=logs/%j.out.log
#SBATCH --error=logs/%j.err.log
#SBATCH --job-name=test

export PATH="/network/lustre/iss02/home/annahita.sarre/anaconda3/envs/myenv/bin:$PATH"
type python

model_type='rf' # 'rf', 'lr'
gender='female' # male/female
cropping='cropped' # raw/cropped
path2test_videos='../stimuli/words/mp4'


python predict.py --property-type position --gender $gender --cropping $cropping --path2test-videos $path2test_videos --fn-video $1 --model-type $model_type
python predict.py --property-type shape --gender $gender --cropping $cropping --path2test-videos $path2test_videos --fn-video $1 --model-type $model_type
python mark_video.py --gender $gender --cropping $cropping --path2video $path2test_videos --fn-video $1 --model-type $model_type --textgrid
