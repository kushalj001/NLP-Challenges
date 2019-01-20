import nltk
import numpy as np
import pandas as pd
import re
import string

n = int(input())
input_text = str(input())

sentences = nltk.sent_tokenize(input_text)

def get_sentences_with_blanks(input_text):
    '''Extracts sentences with blanks from the entire corpus. '''

    #nltk.download('punkt')
    sentences = nltk.sent_tokenize(input_text)
    docs = [sent for sent in sentences if ' ---- ' in sent ]
    new_docs = []
    for doc in docs:
        window = re.findall('[a-zA-Z,\']+ ---- [a-zA-Z,\']+',doc)
        if len(window)>1:
            for i in range(len(window)):
                new_docs.append(doc)
        else:
            new_docs.append(doc)

    return new_docs


def tag_docs(new_docs):
    '''Tags the docs/sentences which have blanks with various parts-of-speech tags in nltk.'''
    #nltk.download('averaged_perceptron_tagger')
    tokenized_docs = []
    tagged_docs = []
    for doc in new_docs:
        tokens = nltk.word_tokenize(doc)
        tokenized_docs.append(tokens)
        tagged_docs.append(nltk.pos_tag(tokens))

    return tagged_docs


def find_verbs_nouns(tagged_docs):
    '''Specifically extracts nouns and verb tags from the previously tagged documents.
       Verb tags are going to be used to extract tense information from the sentence whereas noun tags will be
       used to find whether the subjects in the sentence are plural or singular.'''

    verb_tags_list = []
    noun_tags_list = []
    for doc in tagged_docs:
        tags = ' '.join([tag[1] for tag in doc])
        verb_tags = re.findall('(VB\w?)',str(tags))
        verb_tags_list.append(verb_tags)
        noun_tags = re.findall('(NN\w*)',str(tags))
        noun_tags_list.append(noun_tags)
        #print(verb_tags)
        #print(noun_tags)

    return verb_tags_list,noun_tags_list

def determine_tense(verb_tags_list):
    '''Classifies a sentence based on tense.The tense is determined by counting the number of occurences of each
    type of tense in the sentence using the POS tags.
    Returns a dictionary of tuples with counts of (past,present) verbs respectively'''

    tense_data = {}
    i = 1
    for tags in verb_tags_list:
        present = 0
        past = 0

        for tag in tags:
            if tag == 'VBD' or tag == 'VBN':
                past += 1
            elif tag == 'VBG' and past>0:
                past += 1
            elif tag == 'VB' or tag == 'VBP' or tag == 'VBZ':
                present += 1
            elif tag == 'VBG' and present>0:
                present += 1

        tense_data[i] = (past,present)
        i += 1

    return tense_data

def determine_singular(noun_tags_list):
    '''Returns a dictionary of tuples with counts of (singular,plural) nouns respectively'''
    noun_data = {}
    j = 1
    for tags in noun_tags_list:
        singular = 0
        plural = 0

        for tag in tags:
            if tag in ['NN']:
                singular += 1
            elif tag in ['NNS']:
                plural += 1

        noun_data[j] = (singular,plural)
        j += 1

    return noun_data

def get_context_info(tense_data,noun_data):
    '''Combines the information of tense and noun of a sentence into one nested dictionary as context info.
    The nested dictionary consists n dictionary with each having two keys (past,singular) with boolean values.'''

    context_data = {}
    for k,v in tense_data.items():
        cont_dict = {}
        if v[0]>v[1]:
            cont_dict['past'] = True
        else:
            cont_dict['past'] = False

        context_data[k] = cont_dict

    for k,v in noun_data.items():

        if v[0]>=v[1]:
            context_data[k]['singular'] = True
        else:
            context_data[k]['singular'] = False

    return context_data


def before_and_after_blank_info():
    '''Gets the pos tags for words before and after the blank.This information will be used as the primary base for
    determining the answer. If some information is missing related to tense or pluarality, context_dict will be referred.'''
    docs = [sent for sent in sentences if ' ---- ' in sent ]
    k=1
    before_blank = []
    after_blank = []
    data = {}
    for doc in docs:
        windows = re.findall('[a-zA-Z,\']+ ---- [a-zA-Z,\']+',doc)
        #print(windows)
        for window in windows:
            window_tokens = re.split(' ---- ',window)
            tagged = nltk.pos_tag(window_tokens)
            before_blank.append(tagged[0])
            after_blank.append(tagged[1])

            #print(tagged)
            tag_dict = {}
            for tags in tagged:
                if tags[1] in ['NN','NNP']:
                    tag_dict['singular'] = True
                elif tags[1] in ['NNS','NNPS']:
                    tag_dict['singular'] = False
                elif tags[1] in ['VBD','VBN']:
                    tag_dict['past'] = True
                elif tags[1] in ['VB','VBZ']:
                    tag_dict['past'] = False


            data[k] = tag_dict
            k+=1

    return data,after_blank,before_blank


new_docs = get_sentences_with_blanks(input_text)
tagged_docs = tag_docs(new_docs)
verb_tags_list,noun_tags_list = find_verbs_nouns(tag_docs(new_docs))
tense_data = determine_tense(verb_tags_list)
noun_data = determine_singular(noun_tags_list)
context_data = get_context_info(tense_data,noun_data)

data,after_blank,before_blank = before_and_after_blank_info()

# combine data and context_data
for k,v in data.items():
    if 'singular' not in data[k].keys():
        data[k]['singular'] = context_data[k]['singular']
    elif 'past' not in data[k].keys():
        data[k]['past'] = context_data[k]['past']


answers = {}
#print(data)
for i in range(0,len(before_blank)):
    if before_blank[i][0] == 'I':
        answers[i+1] = 'am'
    elif before_blank[i][1] == 'MD':
        answers[i+1] = 'be'
    elif before_blank[i][0] in ['has','have','had']:
        answers[i+1] = 'been'
    elif before_blank[i][0] in ['is','was','were','am']:
        answers[i+1] = 'being'
    elif (after_blank[i][1] in ['VBG'] and data[i+1]['singular'] == True and data[i+1]['past'] == True) or (after_blank[i][1] in ['VBD','VBN'] and data[i+1]['singular'] == True) or (data[i+1]['past'] == True and data[i+1]['singular'] == True):
        answers[i+1] = 'was'
    elif (after_blank[i][1] in ['VBG'] and data[i+1]['singular'] == False and data[i+1]['past'] == True) or (after_blank[i][1] in ['VBD','VBN'] and data[i+1]['singular'] == False) or (data[i+1]['past'] == True and data[i+1]['singular'] == False):
        answers[i+1] = 'were'
    elif (after_blank[i][1] in ['VBG'] and data[i+1]['singular']==True) or (data[i+1]['past'] == False and data[i+1]['singular'] == True):
        answers[i+1] = 'is'
    elif (after_blank[i][1] in ['VBG'] and data[i+1]['singular']==False) or (data[i+1]['past'] == False and data[i+1]['singular'] == False):
        answers[i+1] = 'are'

        
for k,v in answers.items():
    print(v)
