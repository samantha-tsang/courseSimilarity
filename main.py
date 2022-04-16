import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer

course_des = pd.read_excel("/Users/samanthatsang/PycharmProjects/140xpProject/course catalog data.xlsx")
requisites = pd.read_excel("/Users/samanthatsang/PycharmProjects/140xpProject/requisites data.xlsx")
description = pd.DataFrame(course_des["crs_desc"])

# class type pattern: Starts with captial letter, non greedy match until first comma
class_type_pattern = "(^[A-Za-z]*?), "
class_type = []
for i in range(len(course_des)):
    # try/except runs code under "try" part, and if there is an error it will run the code under "except"
    try:
        class_type.append(pd.Series(course_des["crs_desc"].to_numpy()[i]).str.extract(class_type_pattern).loc[0, 0])
        description.loc[i, "crs_desc"] = re.sub(class_type_pattern, "", description.loc[i, "crs_desc"])
    except:
        # if the course data is "nan" in the original data set, put no course description
        class_type.append("no course desciption")
        print(i)

# hour patter: comma, followed by word/number followed by word hour and maybe "s"
hour_pattern = ", ([a-z1-9].*? hours?)"
hours = []
for i in range(len(course_des)):
    try:
        # takes first capture group
        hours.append(pd.Series(course_des["crs_desc"].to_numpy()[i]).str.extract(hour_pattern).loc[0, 0])
        description.loc[i, "crs_desc"] = re.sub(hour_pattern, "", description.loc[i, "crs_desc"])
    except:
        # if the course data is "nan" in the original data set, put no course description
        hours.append("no course description")
        print(i)

# pre req pattern: starts match with one of [.:;,] (which is always after the hours part of the description), followed
# by the word requisite either with upper or lower case r, the courses, and ends with a capital letter (that is at the end of
# the course number)
requisites_pattern = "[.:;,]? ([rR]equisites?: course[s]? .*?[A-Z]?)\."
pre_req = []
for i in range(len(course_des)):
    try:
        pre_req.append(pd.Series(course_des["crs_desc"].to_numpy()[i]).str.extract(requisites_pattern).loc[0, 0])
        description.loc[i, "crs_desc"] = re.sub(requisites_pattern, "", description.loc[i, "crs_desc"])
    except:
        # if the course data is "nan" in the original data set, put no course description
        pre_req.append("no course desciption")
        print(i)

other_pattern = "^(\(.*?\)) "
for i in range(len(course_des)):
    try:
        description.loc[i, "crs_desc"] = re.sub(other_pattern, "", description.loc[i, "crs_desc"])
    except:
        print(i)

parsed_course_desc = pd.concat([course_des, pd.DataFrame(class_type), pd.DataFrame(hours), pd.DataFrame(pre_req),
                                description], axis=1)
parsed_course_desc.columns = ['srs_crs_no_6', 'subj_area_cd', 'crs_catlg_no', 'crs_career_lvl_cd',
                            'univ_req_cd', 'spcl_prog_cd', 'subttl_req_fl', 'health_sci_fl',
                            'impacted_crs_fl', 'max_alw_atm_unt', 'max_repeat_pn_unt',
                            'crs_mat_fee_amt', 'xlist_id', 'concurrent_id', 'mult_term_grp_id',
                            'mult_term_seq_num', 'crs_short_ttl', 'crs_long_ttl', 'crs_act_typ_cd',
                            'crs_grd_typ_cd', 'crs_unt_typ_cd', 'crs_desc', 'sr_div_cd',
                            'sr_div_name', 'sr_dept_cd', 'sr_dept_name', "class_type", "hours", "pre_req", "clean_desc"]

parsed = parsed_course_desc.copy()

### eliminate courses with no description
parsed = parsed[~parsed["clean_desc"].isnull()]
parsed = parsed.reset_index(drop=True)


# get rid of symbols
def remove_symbols(string):
    string = re.sub("\n", " ", string)  # replace \n symbol with space
    new_string = re.sub("[^a-zA-Z0-9 ]", "", string)  # take away all symbols
    return new_string

remove_symbols(parsed["clean_desc"][0])  # try function
parsed["clean_desc"] = parsed["clean_desc"].map(remove_symbols)

# function to remove stopwords
stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    # input string
    # output: list of tokenized words in string with stop words removed
    sentence = text.split()
    filtered_sentence = [w for w in sentence if w.lower() not in stop_words]
    filtered_sentence = TreebankWordDetokenizer().detokenize(filtered_sentence).lower()
    return filtered_sentence


remove_stopwords(parsed["clean_desc"][0])  # test function
parsed["clean_desc"] = parsed["clean_desc"].map(remove_stopwords)

# save df
parsed.to_csv("/Users/samanthatsang/PycharmProjects/140xpProject/parsed1.csv")

