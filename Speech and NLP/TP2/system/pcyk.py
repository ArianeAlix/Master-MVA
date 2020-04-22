import numpy as np
from nltk import Tree

class PCYK():
    """
    Implements a probabilistic version of the Cocke-Younger-Kasami
    algorithm
    """
    
    def __init__(self,grammar,lexicon,oov,tag2id, id2tag):
        
        self.grammar = grammar # (PCFG grammar)
        self.lexicon = lexicon # (PCFG lexicon)
        self.oov = oov # (Out-of-Vocabulary methods)
        
        self.tag2id = tag2id
        self.id2tag = id2tag
        
    
    def decode_result(self,start,stop,sentence,backtracking,tag_id):
        """
        Recursively builds the tree corresponding to the sentence 
        using the backtracking table built by the PCYK algorithm
        """
        result = ""
        
        
        if start==stop:
            # Reaching terminal word
            result += '(' + self.id2tag[tag_id] +' ' + sentence[int(start)] + ')'
        else:
            split_id, B, C =  backtracking[start,stop,tag_id]

            # Backtrack recursively on the left and right substrings
            lhs = self.decode_result(int(start),int(split_id),sentence,backtracking,int(B))

            rhs = self.decode_result(int(split_id+1),int(stop),sentence,backtracking,int(C))
            
            result += '(' + self.id2tag[tag_id] + ' ' + lhs + ' ' + rhs + ')'
            
        return result
        
        
    def parser(self,sentence):
        """
        Returns a string of the result in bracketed format
        """
        
        # Format sentence as a list of words
        sentence = sentence.strip().split(' ')
        n = len(sentence)
        N = len(self.tag2id.keys()) # Nb of distinct tags
        
        ### Initialization of tables used to store the scores, and 
        # the best rules of the parsed trees obtained during 
        # Dynamic Programming
        values = -1 * np.ones((n, n, N))
        backtracking = np.zeros((n, n, N,3))
        for i in range(n):
            backtracking[i,:,:,0]= i * np.ones((n,N))
            
        for i in range(0,n):
            word = sentence[i]
            
            
            # Check if word in training data
            # else use oov to assign tag using similar words
            if word in self.lexicon.columns:
                tags_proba = np.zeros((N,))
                
                for tag_id in self.id2tag.keys():
                    try:
                        tags_proba[tag_id] = self.lexicon[word][self.id2tag[tag_id]]
                    except:
                        tags_proba[tag_id] = 0
            else:
                tag = self.oov.assign_token(word)
                # Find corresponding index of the tag and 
                # one-hot-encode it as probas like the training words
                tag_id = self.tag2id[tag]
                
                tags_proba = np.zeros((N,))
                tags_proba[tag_id] = 1.0
            
            # Values initialized to the part-of-speech tag probas
            values[i,i] = tags_proba
        
        
        
        ### Iterations over the substrings and backtracking
        # To that end we need only the binary rules of the PCFG
        binary_rules =[]
        rules = self.grammar.productions()
        for rule in rules:
            if len(rule)==2:
                binary_rules.append(rule)
        
        for l1 in range(0,n): # possible lengths of substrings
            for l2 in reversed(range(0,l1)):
                for rule in binary_rules:
                    for split_id in range(l2,l1):

                        # Probability of the split given the rules (We look in both order of rhs rules)
                        proba = rule.prob() * ((values[l2, split_id, self.tag2id[str(rule._rhs[0])]]* values[split_id+1, l1, self.tag2id[str(rule._rhs[1])]])   )
    
                            
                        # Check if not positive because multiplying the initialized -1
                        if values[l2, split_id, self.tag2id[str(rule._rhs[0])]]>=0 and values[split_id+1, l1, self.tag2id[str(rule._rhs[1])]]>=0:
                            if values[l2, l1,self.tag2id[str(rule._lhs)]] < proba:
                                values[l2, l1,self.tag2id[str(rule._lhs)]] = proba
                                # Stores the best split for the substring
                                backtracking[l2, l1,self.tag2id[str(rule._lhs)]] = [split_id, self.tag2id[str(rule._rhs[0])], self.tag2id[str(rule._rhs[1])]]
        
        # Chooses the first tag to start with and encodes
        starting_probas = values[0, n-1]

        while True:
            starting_tag = np.argmax(starting_probas)
            # Retrying if not working with that starting tag
            try:
                if np.sum(starting_probas==-1) != N:
                    result = self.decode_result(0,n-1,sentence,backtracking,starting_tag)
                    
                    # Format it as a simple nltk Tree stored in a string
                    tree_result = Tree.fromstring('(SENT '+result+')')
                    tree_result.un_chomsky_normal_form()
            
                    string_result = str(tree_result).split()
                    string_result = ' '.join(word for word in string_result)
            
                    return '( '+string_result+')'
                
                # If all starting tags have been tried without working
                else:
                    return None
            except:
                # If we had an error with the starting tag,
                # we don't consider it anymore
                starting_probas[starting_tag]=-1
                pass






