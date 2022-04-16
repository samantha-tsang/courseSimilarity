#### Code for calculating course similarity

import pandas as pd
import numpy as np
import re
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

### import dataset
parsed = pd.read_csv("parsed1.csv")

######### option1: tokenize data with detection of bigram #########

### 1. preprocess words in document

# 1.1 get rid of course-type words and any pre-req course numbers
def course_type_remove(desc):
    desc = re.sub("^lecture ", "", desc)
    desc = re.sub("^discussion ", "", desc)
    desc = re.sub("^quiz ", "", desc)
    desc = re.sub("^seminar ", "", desc)
    desc = re.sub("^recitation ", "", desc)
    desc = re.sub("^laboratory ", "", desc)
    desc = re.sub("^studio ", "", desc)
    desc = re.sub("^activity ", "", desc)
    desc = re.sub("^clinic ", "", desc)
    desc = re.sub("^field ", "", desc)
    desc = re.sub("^tutorial ", "", desc)
    desc = re.sub("^research ", "", desc)
    return desc
def course_num_remove(desc):
    return re.sub("[0-9]{2,3}[a-zA-Z]?", "", desc)
def course_hours_remove(desc):
    return re.sub("^ ?[a-zA-Z]{1,6} hours? ", "", desc)

parsed["clean_desc"] = parsed["clean_desc"].map(course_type_remove)
parsed["clean_desc"] = parsed["clean_desc"].map(course_num_remove)
parsed["clean_desc"] = parsed["clean_desc"].map(course_hours_remove)


# 1.2 lemmatize and stem words (ex. change past tense to present, reduce all words to root)
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer(language='english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text): #tokenizes words
        if len(token) > 3: #only take words > 3 letters
            result.append(lemmatize_stemming(token))
    return result
processed_desc = parsed["clean_desc"].map(preprocess)
parsed["processed_desc"] = processed_desc

# 1.3 create bigram phrases
from gensim.models.phrases import Phrases, Phraser
phrases = Phrases(processed_desc)
bigram = Phraser(phrases)
tokenize_doc_bigram = []
for doc in processed_desc:
    tokenize_doc_bigram.append(bigram[doc])
parsed["processed_desc"] = tokenize_doc_bigram
parsed.to_csv("parsed1.csv")

#1.4 create tagged dictionary for model, where tag is the unique course number
tagged_data_bigram = []
for i in range(len(parsed)):
    doc_id = i
    doc = tokenize_doc_bigram[i]
    tagged_data_bigram.append(TaggedDocument(doc, tags=[doc_id]))
parsed["Tag ID"] = range(0, 14729) #add column to represent corresponding tag id for each document
parsed.to_csv("parsed1.csv")


### 2.train and tune model

# 2.1 set range of hyperparameters to test out
vec_size = [15, 30, 45, 60, 80, 110] #defines the size of each document vector
window = [3,4,5] #defines how many context words to use to train word2vec embeddings
dm = [1,0] #defintes whether model uses pv-dm (1) or pv-dbow (0)
combo_num = 1
for i in vec_size:
    for j in window:
        for method in dm:
            count = 0
            model= Doc2Vec(tagged_data_bigram, dm = method,  vector_size = i, window = j, min_count = 1,
                           seed = 1000, workers = 3, epochs = 100)
            model.save("model_vec_" + str(i) + "_window_" + str(j) + "_method_" + str(method) + ".model")
            print(combo_num)
            combo_num += 1
model_results = {"vec_size" : [],
                 "window" : [],
                 "method" : [],
                 "div_percent" : []}

#2.2 test different model for best parameters. test metric: how many of the top ten similar are in the same division
combo_num = 1
for i in vec_size:
    for j in window:
        for method in dm:
            model = Doc2Vec.load("model_vec_" + str(i) + "_window_" + str(j) + "_method_" + str(method) + ".model")
            count = 0
            for index in range(len(course_desc)):
                doc = tagged_data_bigram[i][0]
                similar_top_ten = model.dv.most_similar(index, topn = 10)
                for top in range(0, 10):
                    if course_desc.loc[similar_top_ten[top][0], "sr_div_cd"] == course_desc.loc[index, "sr_div_cd"]:
                        count += 1
            percent = count / (10 * 14500)
            model_results["vec_size"].append(i)
            model_results["window"].append(j)
            model_results["method"].append(method)
            model_results["div_percent"].append(percent)
            print(str(combo_num) + "/36")
            combo_num += 1
pd.DataFrame(model_results) # look at model results to find models with highest correct division percentage. chose top 9
                            # (above 50%)

# 2.3 further look at top models and inspect manually to see which one produces most reasonable results
# pick random set of documents
from random import sample
sample_doc = sample(range(0, 14500), 25)
vec_size = [60,80,110]
window = [3,4,5]
sample_similarity = pd.DataFrame() # create df to store similar doc results
for vec in vec_size:
    for w in window:
        model = Doc2Vec.load("model_vec_" + str(vec) + "_window_" + str(w) + "_method_0.model")
        top_ten = [vec]
        similar = model.dv.most_similar(sample_doc[11], topn=10)
        for i in range(10):
            top_ten.append(similar[i][0])
        sample_similarity = pd.concat([sample_similarity, pd.DataFrame(top_ten)], axis = 1)
# use above code to create a dataframe of top-ten similars (each column represents a method), then manually compare with
# target document to access which is closer
parsed.loc[1233, "crs_desc"] #shows target course description
parsed.loc[325, "crs_desc"] #shows similar course description

# 2.4 top 3 models: vec 110, window 5; vec 110, window 4; vec 80, window: 3
# picked one as final
final_similarity_model = Doc2Vec.load("model_vec_110_window_5_method_0.model")
final_similarity_model.save("final_similarity.model")

# 3 Drawbacks of model:
# not reproducible: random vectors are chosen everytime for negative sampling, although results are generally similar
# model uses bi-gram, future groups can possibly test out uni-gram or tri-gram to compare difference in performance
# metric for tuning is subjective, future group can discuss on possible better ways of tuning it











