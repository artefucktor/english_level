#!/usr/bin/env python
# coding: utf-8

import os
import time

import streamlit as st
import pandas as pd

import pysrt
import chardet

from joblib import load
from english_level_process import ProcessData

ENGLISH_LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
EXAMPLE_SUBS   = 'Fear.2023.720p.WEBRip.x264.AAC-[HQCINEMAS.COM].srt'
DIRNAME = os.path.dirname(__file__)
level = 1

# загружаем модель и класс с предобработкой
model = load(DIRNAME + '/english_level_model.joblib')
process = ProcessData()

# шапка страницы
st.set_page_config(page_title='Уровень английского в субтитрах', 
                   page_icon='📺', 
                   layout="centered", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

with st.container():
    st.header('Определим уровень английского')
    st.markdown('*по субтитрам* 📺 *в фильме*')
    
uploaded_file = st.file_uploader('Загрузите файл с субтитрами в формате .srt', type='.srt')

if uploaded_file is not None:
    # если файл загружен, то определяем кодировку, декодируем и передаем в pysrt
    encoding = chardet.detect(uploaded_file.getvalue())['encoding']
    content = pysrt.from_string(uploaded_file.getvalue().decode(encoding))
    filename = uploaded_file.name
else:
    # по умолчанию показываем демо-файл
    st.write('Например')
    filename = EXAMPLE_SUBS
    fullpath = os.path.join(DIRNAME,filename)
    encoding = chardet.detect(open(fullpath, "rb").read())['encoding']
    content = pysrt.open(fullpath, encoding=encoding)

# обработка файла и предсказание
st.subheader(filename)
with st.spinner('Рассчитываем...'):
    movies = process.process_data(content)
    if len(movies)>0:
        level  = model.predict(movies)[0]
    else:
        filename = EXAMPLE_SUBS


'---'
# расшифровываем предикт в юзер-френдли форму и выводим на страницу
current_level = ENGLISH_LEVELS[min(round(level)-1, len(ENGLISH_LEVELS)-1)]
floor_level   = ENGLISH_LEVELS[min(int(level)-1, len(ENGLISH_LEVELS)-1)]
ceil_level    = ENGLISH_LEVELS[min(int(level), len(ENGLISH_LEVELS)-1)]
sub_level     = '+' if 0.2 <=level%1<0.5 or level>len(ENGLISH_LEVELS)-1 \
                    else '+/'+ceil_level if 0.5<=level%1<0.8 \
                    else ''
user_level    = floor_level + sub_level

st.subheader('Уровень сложности')
st.write(f'Для оценки мы используем опыт специалистов и словари Oxford. '
         f'Наши преподаватели рекомендуют фильм для изучения языка на уровне '
         f'{user_level}.')
st.title(user_level)

level_bar = st.progress(0)
for i in range(min(round(1/6 * level * 100),101)):
    level_bar.progress(i)
    time.sleep(0.001)


'---'
# подсчитаем уровни слов по оксфорду
words_under_level = movies[[l + 'ratio' for l in ENGLISH_LEVELS[:min(round(level),5)]]].sum(axis=1)[0]
words_upper_level = movies[[l + 'ratio' for l in ENGLISH_LEVELS[round(level):5]]].sum(axis=1)[0]

st.subheader('Стандарт CEFR')
st.write('В словарях Oxford 3000 и 5000 — наиболее важные слова, которые должен знать каждый, кто учит английский.')
st.write('Словарь Oxford соответствует стандартам CEFR – The Common European Framework of Reference for Languages. '
         'Это общеевропейские компетенции владения иностранным языком: изучение, преподавание, оценка — '
         'система уровней владения иностранным языком, используемая в Европейском Союзе.')
st.write('Мы рассчитали, слова какого уровня чаще всего встречаются в фильме:')

if words_under_level > 0:
    st.progress(words_under_level, text=f"{words_under_level:.0%} слов до {current_level} уровня включительно")

if words_upper_level > 0:
    st.progress(words_upper_level, text=f"{words_upper_level:.0%} слов превышают уровень {current_level} по сложности")

st.write('остальные слова не имеют определенного уровня.')
    
'---'
# подсчитаем длину слов
less_than_4 = movies.less_than_4ratio[0]
between_4_6 = movies[['len_equal_' + str(i) + 'ratio' for i in range(4,7)]].sum(axis=1)[0]
more_than_6 = movies.more_than_6ratio[0]

st.subheader('Запас слов')
st.write(f'Всего в фильме встречается {movies.lemmas_unique[0]} разных слов.')
st.write('В целом герои фильма испольуют слова вот такой длины:')

if less_than_4 > 0:
    st.progress(less_than_4, text=f"Короткие слова не более трех букв: {less_than_4:.0%}")

if between_4_6 > 0:    
    st.progress(between_4_6, text=f"Средние слова от четырех до шести букв: {between_4_6:.0%}")
    
if more_than_6 > 0:
    st.progress(more_than_6, text=f"Длинные слова от семи и больше букв: {more_than_6:.0%}")

'---'
# подвал
input_text = st.text_area('Было полезно?', placeholder='Поделитесь предложениями, пожеланиями, замечаниями...')
if st.button("Отправить"):
    st.write('На самом деле тут никакого действия не запланировано 😎')
    st.markdown('- Пишите письма [в телеграме](https://t.me/artefucktor)')
    st.markdown('- или заходите [на гитхаб](https://github.com/artefucktor)')



