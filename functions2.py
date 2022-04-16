### 2.1 code for output for user input phrases

import pandas as pd
import numpy as np
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phrases, Phraser

# add column with collapsed subject area code and catalog number
subj_cat = []
for i in range(len(parsed)):
    num = parsed.loc[i, "crs_catlg_no"]
    for j in range(5):
        num = re.sub("^0", "", num)
    subj_cat.append(parsed.loc[i, "subj_area_cd"] + " " + num)
parsed["subj_cat"] = subj_cat

#import preprocessed data set
parsed = pd.read_csv("parsed.csv")
#load model
model = Doc2Vec.load("final_similarity.model")
phrases = Phrases(parsed["processed_desc"])
bigram = Phraser(phrases)

def phrase_find_similar(phrase):
    #tokenize input phrase and remove stop words
    phrases = []
    for token in gensim.utils.simple_preprocess(phrase):
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            phrases.append(token)
    bigram_doc = bigram[phrases]

    #find most similar courses
    similar = model.dv.most_similar(positive=[model.infer_vector(bigram_doc)], topn=len(parsed))
    rank = []
    scores = []
    for i in range(len(parsed) - 1):
        rank.append(similar[i][0])
        scores.append(similar[i][1])

    # create new df where rows are re-ordered according to similarity, re index after
    ranked_parsed = parsed.reindex(rank)
    ranked_parsed["Similarity Score"] = scores
    ranked_parsed = ranked_parsed.reset_index(drop=True)

    sim_courses = []
    for i in range(10):
        sim_courses.append("Course: " + ranked_parsed.loc[i, "subj_area_cd"] + " " +
                           ranked_parsed.loc[i, "crs_catlg_no"] + "; Similarity Score: " +
                           str(ranked_parsed.loc[i, "Similarity Score"]))

    return sim_courses

#test case
phrase_find_similar("linear model")
