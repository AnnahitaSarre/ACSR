
## CREATE TRAIN VIDEOS FROM WORD VIDEOS

First add shape and position information into the event files:
`python add_shape_and_pos_to_events_files.py`

Then, create the videos:
`create_training_videos_from_word_videos.py`



## To run the pipeline, launch:
`./pipeline_full.sh`

Or, for evaluation only, 

`./pipeline_eval_only.sh`
