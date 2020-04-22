# MVA Course "Algorithms for speech and language processing", 2020

### Ariane Alix
### March 16th, 2020

### Importation of libraries and files
import os 
import random

"""
import PYEVALB.scorer as evalscorer
import PYEVALB.parser as evalparser
"""

import time
import argparse
import pickle
import numpy
import pandas
import nltk
import re

from importlib import reload

import process 
import pcfg 
import oov 
import pcyk 

# Re-import modules, useful if they have been changed
reload(process)
reload(pcfg)
reload(oov)
reload(pcyk)

from process import *
from pcfg import *
from oov import *
from pcyk import *


### Parse arguments when executing run.sh
args = argparse.ArgumentParser( description='Basic run script for the Parser' )

args.add_argument( '--input_filename', type=str, required=True, help='Input text file to parse' )
args.add_argument( '--output_filename', type=str, required=True, help='Output file with the results of the parsing')

args = args.parse_args()



### Loading embeddings from Polyglot
print("Loading Polyglot embeddings...")
vocab, embeddings = pickle.load(open('./data/polyglot-fr.pkl', 'rb'), encoding='latin1')

print("Nb of words:",embeddings.shape[0])
print("Length of the embeddings:",embeddings.shape[1])



### Loading and processing of the corpus data
print("\nLoading and processing the SEQUOIA corpus...")
start = time.time()
f = open('sequoia-corpus+fct.mrg_strict', "r", encoding='utf-8')
labeled_sentences = f.readlines()
f.close()



# Splitting into train/dev/test sets
n_train = int(0.8*len(labeled_sentences))
n_dev = int(0.1*len(labeled_sentences))


# Processing to remove the functionnal labels
raw_train = remove_labels(labeled_sentences[:n_train])
raw_dev = remove_labels(labeled_sentences[n_train:n_train+n_dev])
raw_test = remove_labels(labeled_sentences[n_train+n_dev:])

labeled_train = remove_nonterminal_labels(labeled_sentences[:n_train])
labeled_dev = remove_nonterminal_labels(labeled_sentences[n_train:n_train+n_dev])
labeled_test = remove_nonterminal_labels(labeled_sentences[n_train+n_dev:])

print('Time taken:',round(time.time()-start,3),'s')
 
 

### Create PCFG from the training sentences
print("\nGenerating Probabilistic Context-Free Grammar from training data...")
start = time.time()
cfg = PCFG(labeled_train)
grammar = cfg.get_build_grammar()

        

lexicon = cfg.get_build_lexicon()
tag2id, id2tag = cfg.get_build_tags_dicts()



print("Lexicon shape (nb of tags, nb of tokens):",lexicon.shape)

print('Time taken:',round(time.time()-start,3),'s')



### Create Out-Of-Vocabulaty module
oov=OOV(vocab,embeddings,lexicon)


### Create PCYK

cyk = PCYK(grammar,lexicon,oov,tag2id, id2tag)

        
        
### Examples
print("\nParsing example from training set: '"+raw_train[1]+"'")
result1 = cyk.parser(raw_train[1]) 
print("Result:",result1)
  
print("\nParsing invented example: 'Gutenberg est dans les champs'")
result2 = cyk.parser("Gutenberg est dans les champs") 
print("Result:",result2)
  

### Apply on evaluation dataset if wanted 
evaluation = False
if evaluation:
    n_test = len(labeled_test)
    results = []
    for i in range(n_test):
        sentence = raw_test[i]
        result = cyk.parser(sentence)
        results.append(result)
        
    f1 = open('./evaluation_data.parser_output', 'a')
    for result in results:
        if result is not None:
            f1.write(result)
        else:
            f1.write('Parsing failed')
        f1.write('\n')
    f1.close()



### Parse sentences from input file and store result

print("\nLoading and processing the sentences from the input file...")
start = time.time()
f = open(args.input_filename, "r", encoding='utf-8')
input_sentences = f.readlines()
f.close()


n = len(input_sentences)
results = []
for i in range(n):
    sentence = input_sentences[i]
    result = cyk.parser(sentence)
    results.append(result)
    
f1 = open(args.output_filename, 'a')
for result in results:
    if result is not None:
        f1.write(result)
    else:
        f1.write('Parsing failed')
    f1.write('\n')
f1.close()

print("Results saved to",args.output_filename)



### Scoring of accuracy
# Takes too long so not launched

"""
def test(cyk, raw_sentence, labeled_sentence, scorer):
    result = cyk.parser(raw_sentence)
    if result is not None:
        pred_tree = evalparser.create_from_bracket_string(result)
        real_tree = evalparser.create_from_bracket_string(labeled_sentence)
        score = scorer.score_trees(real_tree, pred_tree)
    else:
        score = evalscorer.Result()
        score.state = 2
    return score



results = []

n = len(raw_test)



for i in range(n):
    scorer = evalscorer.Scorer()
    raw_sentence = raw_test[i]
    labeled_sentence = labeled_test[i]

    results.append(test(cyk, raw_sentence, labeled_sentence, scorer))



score_summary = evalscorer.summary.summary(results)
print(score_summary)
"""



        
            
            
            
            
            