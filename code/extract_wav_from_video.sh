
#set -x

for i in `seq -w 1 99`
do	
    cmd='python extract_wav_from_video.py --fn-video sent_'$i'.mp4'
    echo $cmd
    eval $cmd
done
 
