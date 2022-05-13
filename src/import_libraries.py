# nltk packages

from nltk.corpus import stopwords
from nltk.collocations import *
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.tokenize import RegexpTokenizer

from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

from nltk import WordNetLemmatizer # lemmatizer using WordNet
from nltk.corpus import wordnet # imports WordNet
from nltk import pos_tag # nltk's native part of speech tagging
import nltk

# sklearn packages

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split,  GridSearchCV

from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import  MaxAbsScaler, StandardScaler
from xgboost import XGBClassifier

from sklearn import set_config
set_config(display ="diagram")

# Other packages

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import string
import re
import nltk
import json
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 30)
pd.set_option('max_colwidth', 800)

import warnings
warnings.filterwarnings("ignore") #Otherwise it complains about depreciated commands


