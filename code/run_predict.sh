path2test_videos='../stimuli/words/mp4'
for fn_video in $path2test_videos/word_*.mp4; do
    sbatch predict_and_mark.sh $(basename -- $fn_video)
done
