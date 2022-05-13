#!/usr/bin/env python

# # Building a Preprocessing Pipe for NLP

from import_libraries import *

######################################################################

# ## Pre-processing functions
# To be used by vectorizers as a part of prep_pipe:


def process_review(review, return_list=False):
    review=review.lower()
    review_norm=word_tokenize(review)
    review_norm  = [SnowballStemmer('english').stem(token) for token in review_norm]
    review_norm = [x for x in review_norm if (x.isalpha() & (x not in stop_words) ) ]

    if return_list:
        return review_norm
    else:
        return " ".join(review_norm)

def process_ngram_review(review):
    review=review.lower()
    words = review.translate(review.maketrans('', '', string.digits+string.punctuation))
    return words


######################################################################

# ## Custom sklearn classes to build New Features

######################################################################

## WordCounter class - creates a column counting number of words for each review

class WordCounter (BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, data,  y = 0):
        return self

    def transform(self, data, y = 0):
        words_n = data.apply(lambda x: len(x.split()) )
        return words_n.values.reshape(-1,1)

######################################################################

## StringCounter class - creates a column counting occurences of any passed string for each review

class StringCounter (BaseEstimator, TransformerMixin):
    def __init__(self, str_to_count):
        self.str_to_count=str_to_count

    def fit(self, data,  y = 0):
        return self

    def transform(self, data, y = 0):
        string_n = data.apply(self.count_string)
        return string_n.values.reshape(-1,1)

    def count_string(self, data):
        string_n=data.count(self.str_to_count)
        total=np.sum([1 for x in data if x.isalpha()])
        if total==0:
            total=1
        string_p=string_n/total
        return string_p

######################################################################

## CapitalCounter class - creates a column counting occurences of Capitalized characters for each review

class CapitalCounter (BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, data,  y = 0):
        return self

    def transform(self, data, y = 0):
        capital_count=data.apply(self.count_capital)
        return capital_count.values.reshape(-1,1)

    def count_capital(self, data):
        capital_n=np.sum([1 for x in data if x.isupper()])
        total=np.sum([1 for x in data if x.isalpha()])
        if total==0:
            total=1
        capital_p=capital_n/total
        return capital_p

######################################################################

## MisspellCounter class - creates a column counting occurences of misspelled words for each review

class MisspellCounter (BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, data,  y = 0):
        return self

    def transform(self, data, y = 0):
        misspell_count=data.apply(self.count_misspell)
        return misspell_count.values.reshape(-1,1)

    def count_misspell(self, data):
        misspell_n=np.sum([1 for x in data.split() if x.isalpha() and webster.get(x,0)==0])
        total=np.sum([1 for x in data.split() if x.isalpha()])
        if total==0:
            total=1
        misspell_p=misspell_n/total
        return misspell_p


######################################################################

# ## Loading webster dictionary and stop_words

# Loading webster dictionary to identify misspelled words

with open('../data/supplementary/webster.json') as data:
    webster = json.load(data)

exceptions=["against", "again", "should've", "should", 'because','few']
stop_words = stopwords.words('english') # + stop_words
for exc in exceptions: stop_words.pop(stop_words.index(exc))


######################################################################

# ## Building a Pre-processing Pipe
#Initialiazing sklearn classes:

text_vectorizer =CountVectorizer(preprocessor=process_review, min_df=0.0004)
bi_vectorizer = CountVectorizer( preprocessor=process_ngram_review, ngram_range=(2, 3), min_df=0.0006)
bi_summ_vectorizer = CountVectorizer(preprocessor=process_ngram_review, ngram_range=(2, 4), min_df=0.0006)

word_counter=WordCounter()
quest_counter=StringCounter('?')
excl_counter=StringCounter('!')
misspell_counter=MisspellCounter()
capital_counter=CapitalCounter()

scaler=MaxAbsScaler()

######################################################################

# FeatureUnion to be used on Text column:

text_fu = FeatureUnion([
    ('word_counter', word_counter),
    ('capital_counter', capital_counter),
    ('quest_counter', quest_counter),
    ('excl_counter', excl_counter),
    ('text_vect', text_vectorizer),
    ('bi_text_vect', bi_vectorizer),
])

# FeatureUnion to be used on Summary column:

summ_fu = FeatureUnion([
    ('misspell_counter', misspell_counter) ,
    ('quest_counter', quest_counter) ,
    ('excl_counter', excl_counter),
    ('capital_counter',  capital_counter),
    ('sum_vect', text_vectorizer),
    ('bi_summ_vect', bi_summ_vectorizer)
])

# ColumnTransformer to combine both FeatureUnions:

preprocessor = ColumnTransformer(transformers=[
    ('text_fu', text_fu, 'Text'),
    ('summ_fu', summ_fu, 'Summary'),
], remainder='passthrough')

######################################################################

# Prep_pipe to combine preprocessor and scaler:

prep_pipe = Pipeline([('prep', preprocessor),  ('scaler', scaler)
                     ]);
# * We have our pre-processing pipeline, now we are ready to try different models


