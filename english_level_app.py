#!/usr/bin/env python
# coding: utf-8

import os
import re
import time

import streamlit as st
import pandas as pd

from joblib import load

from detect_english_level import DetectEnglishLevel

ENGLISH_LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1']
EXAMPLE_SUBS   = 'Fear.2023.720p.WEBRip.x264.AAC-[HQCINEMAS.COM].srt'
level = 1

model = load(os.path.dirname(__file__) + '/model_english_level.joblib')
detector = DetectEnglishLevel()

st.set_page_config(page_title='Уровень английского в субтитрах', 
                   page_icon='📺', 
                   layout="centered", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

with st.container():
    st.header('Определим уровень английского')
    st.markdown('*по субтитрам 📺 в фильме*')
    
uploaded_file = st.file_uploader('Загрузите файл с субтитрами в формате .srt', type='.srt')

if uploaded_file is not None:
    content = uploaded_file.getvalue()
    filename = uploaded_file.name
else:
    file = open(f'{os.path.dirname(__file__)}/{EXAMPLE_SUBS}', 'rb')
    content = file.read()
    file.close()
    st.write('Например')
    filename = EXAMPLE_SUBS

st.subheader(filename)
with st.spinner('Рассчитываем...'):
    subs = detector.process_srt(content)
    if subs:
        movies = detector.process_data(subs)
        level = model.predict(movies)[0]

current_level = ENGLISH_LEVELS[round(level)-1]
sub_level     = '+' if 0.2 < level%1 < 0.5 else ''
'---'
st.subheader('Уровень сложности')
st.write(f'Для оценки мы используем опыт специалистов и словари Oxford. '
         f'Наши преподаватели рекомендуют фильм для изучения языка на уровне '
         f'{current_level + sub_level}.')
st.title(current_level + sub_level)

level_bar = st.progress(0)
for i in range(round(1/6 * level * 100)):
    level_bar.progress(i)
    time.sleep(0.001)


'---'
st.subheader('Стандарт CEFR')
st.write('В словарях Oxford 3000 и 5000 — наиболее важные слова, которые должен знать каждый, кто учит английский.')
st.write('Словарь Oxford соответствует стандартам CEFR – The Common European Framework of Reference for Languages. '
         'Это общеевропейские компетенции владения иностранным языком: изучение, преподавание, оценка — '
         'система уровней владения иностранным языком, используемая в Европейском Союзе.')
st.write('Мы рассчитали, слова какого уровня чаще всего встречаются в фильме:')

words_under_level = movies[[l + 'ratio' for l in ENGLISH_LEVELS[:round(level)]]].sum(axis=1)[0]
words_upper_level = movies[[l + 'ratio' for l in ENGLISH_LEVELS[round(level):]]].sum(axis=1)[0]

if words_under_level > 0:
    st.progress(words_under_level, text=f"{words_under_level:.0%} слов до {current_level} уровня включительно")

if words_upper_level > 0:
    st.progress(words_upper_level, text=f"{words_upper_level:.0%} слов превышают уровень {current_level} по сложности")

st.write('остальные слова не имеют определенного уровня.')
    
'---'
st.subheader('Запас слов')
st.write(f'Всего в фильме встречается {movies.lemmas_unique[0]} разных слов.')
st.write('В целом герои фильма испольуют слова вот такой длины:')

less_than_3 = movies.less_than_3ratio[0]
between_4_6 = movies[['equal_' + str(i) + 'ratio' for i in range(4,7)]].sum(axis=1)[0]
more_than_7 = movies.more_than_7ratio[0]

if less_than_3 > 0:
    st.progress(less_than_3, text=f"Короткие слова не более трех букв: {less_than_3:.0%}")

if between_4_6 > 0:    
    st.progress(between_4_6, text=f"Средние слова от четырех до шести букв: {between_4_6:.0%}")
    
if more_than_7 > 0:
    st.progress(more_than_7, text=f"Длинные слова от семи и больше букв: {movies.more_than_7ratio[0]:.0%}")

'---'
input_text = st.text_area('Было полезно?', placeholder='Поделитесь предложениями, пожеланиями, замечаниями...')
if st.button("Отправить"):
    st.write('На самом деле тут никакого действия не запланировано 😎 '
             'пишите письма [на гитхабе](https://github.com/artefucktor)')



