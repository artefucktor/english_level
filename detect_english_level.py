#!/usr/bin/env python
# coding: utf-8

from joblib import load
import re
import os

import chardet
import spacy

import pandas as pd


class DetectEnglishLevel():
    
    def __init__(self):
        self.ENGLISH_LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1']
        self.oxford_words = load(os.path.dirname(__file__) + '/oxford_dictionary.joblib')
        self.nlp = spacy.load("en_core_web_sm")
    
    def process_srt(self, content):
        # process file
        print('process file')
        try:
            encoding = chardet.detect(content)['encoding']      # detect encoding and convert
            print(encoding)
            content = content.decode(encoding)
        except:
            print('не прочиталось')
            return False
        text = []
        for line in content.splitlines():                        
            if re.search(r'[A-Za-z]',line):                      # get non empty lines with words
                line = line.lower()                              # transform to lowercase
                line = re.sub(r'\<[^<]+?\>', ' ', line)          # remove html tags
                line = re.sub(r'(\(.+\))|(\[.+\])', ' ', line)   # remove actions in parenthesis
                line = re.sub(r'([\w#\s]+\:)', ' ', line)        # remove speaker in dialogs
                line = re.sub(r'(?<=\w)\'(?=\w)', '', line)      # glue apostroph inside words
                line = re.sub(r'[^a-z]', ' ', line)              # remove non-word symbols
    #             line = re.sub(r'\b\w{1,2}\b', '', line)          # remove single and two symbol sequences
                line = re.sub(r'\s\s+', ' ', line).strip()       # remove extra spaces longer than single
                if len(line)>0:
                    text.append(line)
        return ' '.join(text)

    def count_words_by_level(self, row):
        # oxford words in movies lemmas
        row_lemmas = row.lemmas.split()
        total_score = self.oxford_words.loc[self.oxford_words.word.isin(row_lemmas)].groupby('level').word.count().to_dict()
        row[total_score.keys()] = total_score
        return row


    def coverage_oxford(self, row, l, oxford_equal, oxford_less, oxford_more):

        row_lemmas = set(row.lemmas.split())

        cover_col = 'coverage_' + l
        ratio_col = 'coverage_' + l + 'ratio'

        cover_less = 'coverage_less_equal_' + l
        ratio_less = 'coverage_less_equal_' + l + 'ratio'

        cover_more = 'coverage_more_than_' + l
        ratio_more = 'coverage_more_than_' + l + 'ratio'

        # покрытие словаря в фильме (по уровням)
        row[cover_col] = sum(lemma in oxford_equal for lemma in row_lemmas)
        row[ratio_col] = row[cover_col] / len(oxford_equal)

        # покрытие словаря суммарно по уровням до заданного включительно
        row[cover_less] = sum(lemma in oxford_less for lemma in row_lemmas)
        row[ratio_less] = row[cover_less] / len(oxford_less)

        # покрытие словаря выше заданного уровня
        len_oxford_more = len(oxford_more)
        if len_oxford_more>0:
            row[cover_more] = sum(lemma in oxford_more for lemma in row_lemmas)
            row[ratio_more] = row[cover_more] / len(oxford_more)
        else:
            row[cover_more], row[ratio_more] = 0, 0

        return row
    
    def process_data(self, subs):
        
        movies = pd.DataFrame({'content' : subs,
                               'level'   : None
                              }, index=[0])
        for col in self.ENGLISH_LEVELS + [l+'ratio' for l in self.ENGLISH_LEVELS]:
            movies[col] = None
            
        movies['lemmas'] = movies.content.transform(lambda text: ' '.join([w.lemma_ for w in self.nlp(text)]))
        movies = movies.apply(self.count_words_by_level, axis=1)
        for l in self.ENGLISH_LEVELS:
            oxford_equal = self.oxford_words[self.oxford_words.level==l].word.values
            oxford_less  = self.oxford_words[self.oxford_words.level<=l].word.values
            oxford_more  = self.oxford_words[self.oxford_words.level>l].word.values
            movies = movies.apply(self.coverage_oxford, axis=1, args=(l, oxford_equal, oxford_less, oxford_more))
            
        # total lemmas count
        # and unique lemmas count
        # and unique ratio
        movies['lemmas_count']  = movies.lemmas.transform(lambda x: len(x.split()))
        movies['lemmas_unique'] = movies.lemmas.transform(lambda x: len(set(x.split())))
        movies['lemmas_unique_ratio'] = movies.lemmas_unique / movies.lemmas_count
        # ratio oxford words in content
        for l in self.ENGLISH_LEVELS:
            movies[l+'ratio'] = movies[l]/movies.lemmas_unique

        # count words by length
        # and ratio to total count
        for i in range(1,10):
            equal_col = 'equal_'+str(i)
            more_col  = 'more_than_'+str(i)
            less_col  = 'less_than_'+str(i)

            movies[equal_col] = movies.lemmas.transform(lambda x: sum(len(word)==i for word in x.split()))
            movies[more_col]  = movies.lemmas.transform(lambda x: sum(len(word)>i for word in x.split()))
            movies[less_col]  = movies.lemmas.transform(lambda x: sum(len(word)<i for word in x.split()))

            movies[equal_col + 'ratio'] = movies[equal_col] / movies.lemmas_count
            movies[more_col + 'ratio']  = movies[more_col] / movies.lemmas_count
            movies[less_col + 'ratio']  = movies[less_col] / movies.lemmas_count

        movies['target'] = movies.level.replace({'A1':1, 'A2':2, 'B1':3, 'B2':4, 'C1':5, 'C2':6})
        movies = movies.fillna(0)

        return movies

