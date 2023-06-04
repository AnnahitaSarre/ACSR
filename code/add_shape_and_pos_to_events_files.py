import os
import glob
import pandas as pd

from utils import get_word_code


path2stimuli = '../stimuli/words/mp4'
fns_events = glob.glob(os.path.join(path2stimuli, '*.events'))

# ADD SHAPE AND POSITION TO EVENTS FILE
for fn_events in fns_events:
    word = open(fn_events, 'r').readlines()[0].strip('\n')
    df = pd.read_csv(fn_events, skiprows=1, index_col=False)
    
    poss, shapes = [], []
    for i_row, row in df.iterrows():
        if row['event'] == "SYLLABLE ONSET":
            syllable = row['stimulus'].strip()
            word_code = get_word_code(syllable)
            if word_code is not None:
                shape, pos = list(word_code)
            else:
                shape, pos = None, None
        else:
            shape, pos = None, None
        poss.append(pos)
        shapes.append(shape)
    
    df['pos'] = poss
    df['shape'] = shape
    
    df.index.name = word
    
    df.to_csv(fn_events)
    