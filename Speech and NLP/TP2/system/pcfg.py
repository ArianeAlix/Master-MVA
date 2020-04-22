import numpy as np
import pandas as pd
from nltk import Tree, induce_pcfg, Nonterminal

from process import *



class PCFG(object):
    """
    Class defining the Probabilistic Context-Free Grammar
    """
    
    def __init__(self,labeled_training_data):
        
        # Extracts and stores the trees of the sentences
        self.trees = [Tree.fromstring(line, remove_empty_top_bracketing=True) for line in labeled_training_data]
        
        # Translate the trees to Chomsky normal form
        # They now have a left-hand-side and a right-hand-side tag and we can build a Grammar
        for tree in self.trees:
            tree.chomsky_normal_form(horzMarkov=2)
            tree.collapse_unary(collapsePOS=True)
            
        # To store the tags
        self.tags = []
        
        
    def get_build_tags_dicts(self):
        """
        Builds a tag2id and a id2tag dictionnary
        to facilitate CYK computations
        """
        
        # Recursively goes down the tree and stores tags
        def get_tags(tree):
            if len(tree) == 1:
                # Check if terminal
                if type(tree[0]) == str:
                    return None
                else:
                    self.tags.append(tree[0].label())
                    get_tags(tree[0])
                    
            else:
                self.tags.append(tree[0].label())
                self.tags.append(tree[1].label())
                get_tags(tree[0])
                get_tags(tree[1])

    
        for tree in self.trees:
            get_tags(tree)

        # Adding Part-of-speech tags from lexicon
        self.tags = np.unique(list(np.unique(self.tags)) + list(np.unique(self.proba_token_tag.index)) + ['SENT'])
        
        self.tag2id = {tag:id for id,tag in enumerate(self.tags)}
        self.id2tag = {id:tag for id,tag in enumerate(self.tags)}
        
        
        return self.tag2id, self.id2tag
                
        
    def get_build_grammar(self):
        """
        Gets the production rules of the training trees
        and induces a whole grammar using nltk
        """

        prod_rules = []
        for tree in self.trees:
            for prod_rule in tree.productions():
                if not prod_rule.is_lexical():
                    prod_rules.append(prod_rule)
        
        self.grammar = induce_pcfg(Nonterminal('SENT'),prod_rules)
        

        return self.grammar
        
        
        
    def get_build_lexicon(self):
        """
        Builds and returns a probabilistic lexicon, i.e. triples 
        of the form (token, part-of-speech tag, probability) such that
        the sum of the probabilities for all triples for a given token
        sums to 1.
        
        Type of output : dataframe n_tags x n_tokens
        """
        # We are interested in the left-hand-side tag of each token
        # in each tree
        
        #self.count_token_tag = pd.DataFrame(index=self.tags)
        self.count_token_tag = pd.DataFrame({'_init_':0},index=['_init_'])
        
        for tree in self.trees:
            # Get the leaves and their corresponding labels
            n_leaves = len(tree.leaves())
            
            for i in range(n_leaves):
                # Get the position of the leaf
                pos_leaf = tree.leaf_treeposition(i)
                
                # Get the token
                token = tree[pos_leaf]
                
                # Get the corresponding lhs tag
                tag = str(tree[pos_leaf[:-1]].label())
                
                # If there are more levels of tags
                for up in range(2,10):
                    if (len(tree[pos_leaf[:-up]]))==1 and str(tree[pos_leaf[:-up]].label()) not in ['','SENT']:
                        tag = str(tree[pos_leaf[:-up]].label())+'+'+tag
                
                # Adding +1 to the count of tag/token occurrences
                if tag not in self.count_token_tag.index:
                    self.count_token_tag.loc[tag] = 0
                
                if token not in self.count_token_tag.columns:
                    self.count_token_tag[token] = 0
                
                self.count_token_tag.loc[tag,token] += 1
        
        # Remove fake columns/lines added for initialization
        self.count_token_tag = self.count_token_tag.iloc[1:,1:]
        
        
        # Add lines of 0 for other tags
        for tag in self.tags:
            if tag not in self.count_token_tag.index:
                self.count_token_tag.loc[tag] = 0
        
      
        
        # Normalize the probabilities
        self.proba_token_tag = self.count_token_tag / np.sum(self.count_token_tag,axis=0)

        return self.proba_token_tag