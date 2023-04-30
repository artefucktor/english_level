#!/usr/bin/env python
# coding: utf-8

import warnings
import re
import os

import pandas as pd
from joblib import load

import spacy
import pysrt

class ProcessData():
    
    def __init__(self):
        self.ENGLISH_LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1']
        self.oxford_words = load(os.path.dirname(__file__) + '/english_level_oxford.joblib')
        self.nlp = spacy.load("en_core_web_sm")
    
    # process line
    def process_line(self, line):
        if re.search(r'[A-Za-z]',line): 
            line = line.lower()
            line = re.sub(r'\n', ' ', line)                            # remove new lines
            line = re.sub(r'- ', ' ', line)                            # remove dash
            line = re.sub(r'\<[^\<]+?\>', '', line)                    # remove html tags
            line = re.sub(r'\([^\(]+?\)', '', line)                    # remove () parenthesis
            line = re.sub(r'\[[^\[]+?\]', '', line)                    # remove [] parenthesis
            line = re.sub(r'^([\w#\s]+\:)', ' ', line)                 # remove speaker tag
            line = re.sub(r'[^[:alnum:][:punct:][:blank:]]',' ', line) # remove all other non-speach shars
            line = re.sub(r'\s\s+', ' ', line).strip()                 # remove extra spaces
        return line


    # process text line by line   
    def process_text(self, content):
        text = []
        duration  = []
        charsrate = []
        wordsrate  = []
        for item in content:
            try:
                if item.duration.ordinal>0:
                    line = self.process_line(item.text_without_tags)
                    text.append(line)
                    duration.append(item.duration.to_time())
                    charsrate.append(item.characters_per_second)
                    wordsrate.append(len(line.split())/item.duration.ordinal*1000.0)
            except:
                pass
        return ' '.join(text), duration, charsrate, wordsrate


    def count_words_by_level(self, row):
        # oxford words in movies lemmas
        row_lemmas = row.lemmas.split()
        total_score = self.oxford_words.loc[self.oxford_words.word.isin(row_lemmas)].groupby('level').word.count().to_dict()
        row[total_score.keys()] = total_score
        return row
    
    
    def process_data(self, content):
        
        movies = pd.DataFrame()
        result = self.process_text(content)               # process file
        if result:                                        # add movie to dataframe
            subs, duration, charsrate, wordsrate = result
            movies = pd.DataFrame({
                'content'   : subs,
                'duration'  : [duration],
                'charsrate' : [charsrate],
                'wordsrate' : [wordsrate],
            },index=[0])
            movies['charsrate'] = [charsrate]
            movies['wordsrate'] = [wordsrate]
            movies['content']   = subs
            movies['target']    = 0
            for col in self.ENGLISH_LEVELS + [l+'ratio' for l in self.ENGLISH_LEVELS]:
                movies[col] = None
                        
            # lemmatization
            movies['lemmas'] = movies.content.transform(lambda text: 
                                                        ' '.join([w.lemma_.strip() 
                                                                  for w in self.nlp(re.sub(r'[^a-z\s\']', '', text))
                                                                 ])
                                                       )
            
            # total lemmas count
            # and unique lemmas count
            # and unique ratio
            movies['lemmas_count']  = movies.lemmas.transform(lambda x: len(x.split()))
            movies['lemmas_unique'] = movies.lemmas.transform(lambda x: len(set(x.split())))
            movies['lemmas_unique_ratio'] = movies.lemmas_unique / movies.lemmas_count
            
            # count oxford words by levels
            movies = movies.apply(self.count_words_by_level, axis=1)

            # ratio oxford words in content
            for l in self.ENGLISH_LEVELS:
                movies[l+'ratio'] = movies[l]/movies.lemmas_unique

            # count words by length
            # and ratio to total count
            for i in range(1,10):
                equal_col = 'len_equal_'+str(i)
                more_col  = 'more_than_'+str(i)
                less_col  = 'less_than_'+str(i)

                movies[equal_col] = movies.lemmas.transform(lambda x: sum(len(word)==i for word in x.split()))
                movies[more_col]  = movies.lemmas.transform(lambda x: sum(len(word)>i for word in x.split()))
                movies[less_col]  = movies.lemmas.transform(lambda x: sum(len(word)<i for word in x.split()))

                movies[equal_col + 'ratio'] = movies[equal_col] / movies.lemmas_count
                movies[more_col + 'ratio']  = movies[more_col] / movies.lemmas_count
                movies[less_col + 'ratio']  = movies[less_col] / movies.lemmas_count

            movies = movies.fillna(0)

        return movies
