
from import_libraries import *

# Helper function to map pos_tag and wordnet for "get_lemmatized" function

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None


# Function to clean and replace words with its base/dictionary form

def get_lemmatized (review, return_list=False):
    wnl = WordNetLemmatizer()

    exceptions=['because','few','aren', "aren't",  "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
            'haven', "haven't", 'isn', "isn't",  "mightn't",  "mustn't",  "needn't", "shouldn't", 'wasn', "wasn't", 
            'weren', "weren't", 'won', "won't",  "wouldn't","not","but","against", "again", "should've" , "should",
            'very',"don't"]

    stop_words = stopwords.words('english') # + stop_words
    for exc in exceptions: stop_words.pop(stop_words.index(exc))

    review  = review.translate(review.maketrans('', '', string.digits+'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'))
    review_norm=review.lower().split()
      
    # creates list of tuples with tokens and POS tags in wordnet format
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tag(review_norm))) 
    
    review_norm = [x for x in review_norm if (x.isalpha() or re.search ("'", x)) ]
    review_norm = [wnl.lemmatize(token, pos) for token, pos in wordnet_tagged if pos is not None]
    review_norm = [x for x in review_norm if ((x not in stop_words) & (len(x) >3) ) ]

    if return_list:
        return review_norm
    else:
        return " ".join(review_norm)

# function  to create
# 1. frequency dictionaries for each class 
# 2. difference dictionaries of words specific only for each classes.

def bag_of_words(reviews_df, target_col):
    freqdist_dic={}
    diff_dic={'neg':{},'pos':{}}
    voc=list(reviews_df['Text'].str.lower().str.split().explode().unique())
    
 # Creating FreqDist dictionary
    for score in reviews_df[target_col].unique():
        score_df=reviews_df[reviews_df[target_col]==score].copy()
        tokens=score_df['Text'].apply(lambda x: get_lemmatized(x, True))

        freqdist_score= FreqDist(dict(tokens.explode().value_counts(normalize=True)))
        freqdist_dic[str(score)] = freqdist_score

 # Creating  dictinary of differences:
    for word in voc:
        diff=freqdist_dic['pos'].get(word,0)-freqdist_dic['neg'].get(word,0)
        if diff < 0:
            diff_dic['neg'][word]=np.abs(diff)
        elif diff > 0:
            diff_dic['pos'][word]=diff
    return diff_dic, freqdist_dic

######################################
### Visualization functions:
######################################

### Helper function for transform_image
def transform_format(val):
    if val >100: 
        return 255
    else:
        return val
    
### Function to transform an image to fit wordcloud

def transform_image (image):
    thumbs=np.array(Image.open(image).resize((820,820)))[:,:,1]
    transformed_thumb = np.ndarray((thumbs.shape[0],thumbs.shape[1]), np.int32)

    for i in range(len(thumbs)):
        transformed_thumb[i] = list(map(transform_format, thumbs[i]))  
    return transformed_thumb

