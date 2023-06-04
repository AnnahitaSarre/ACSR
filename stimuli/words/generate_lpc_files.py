#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 08:16:49 2022

@author: yair
"""

import os
import glob
import pandas as pd

fn = 'final_words.xlsx' # Change 'words' for 'pseudowords' when dealing with pseudowords
df_words = pd.read_excel(fn)

fns = glob.glob(os.path.join('txt', 'word_*.txt'))
for fn in fns:
    lines = open(fn, 'r').readlines()
    word = lines[0]
    df_word = df_words.query(f'Word=="{word}"') # Change 'Words' for 'pseudo' when dealing with pseudowords
    
    if not df_word.empty:
        word_in_lpc = ' '.join(df_word['phon_lpc'].values[0].split('-'))
        
        fn_lpc = fn[:-4] + '.lpc'
        
        with open(fn_lpc, 'w') as f:
            f.write(word_in_lpc)
    else:
        print(f'WARNING: word not found {word} in {fn}')
