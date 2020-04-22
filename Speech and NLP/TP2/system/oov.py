import numpy as np


def levenshtein_distance(w1, w2):
    """
    Computes the Levenshtein distance between 2 words.
    """
    
    l1 = len(w1)
    l2 = len(w2)
    
    # Initialiation of d the matrix used to compute the distance using
    # Dynamic Programming
    d = np.zeros((l1+1, l2+1))
    
    d[:,0] = np.arange(l1+1)
    d[0,:] = np.arange(l2+1)
    
    for i in range(1,l1+1):
        for j in range(1,l2+1):
            sub_cost = 0
            if w1[i-1] != w2[j-1]:
                sub_cost = 1
            
            # We take the min between the deletion, insertion or substitution costs
            d[i,j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1]+sub_cost)
                
    return d[l1,l2]





class OOV():
    """
    Out-Of-Vocabulary Class that assigns a (unique) part-of-speech 
    to any token not included in the lexicon extracted from 
    the training corpus
    """
    def __init__(self,vocab,fr_embeddings,lexicon):
        # fr_embeddings is the french Polyglot set of word embeddings
        # for the words in vocab
        
        self.vocab = list(vocab)
        self.embeddings = fr_embeddings
        self.lexicon = lexicon
        
        self.word2id = {word:id for id,word in enumerate(vocab)}
        
        
        # Gets a list of the words of the training corpus
        self.words_training = lexicon.columns

    
    
    def cosine_similarity(self,w1,w2):
        """
        Returns a similarity score between 2 words
        based on their embeddings
        """
        # Get embeddings of the words
        e1 = self.embeddings[self.word2id[w1]]
        e2 = self.embeddings[self.word2id[w2]]
        
        # Compute the cosine similarity
        cos_sim = np.dot(e1,e2)/(np.linalg.norm(e1)*np.linalg.norm(e2))
        
        return cos_sim
    
    
    
    def most_similar(self, word, k=5):
        """
        Returns the k most similar words of the training corpus 
        for which we have an embedding
        """
        scores = []
        words = []
        for w in self.words_training:
            # Check if the word has an embedding
            if w in self.word2id.keys():
                scores.append(self.cosine_similarity(word,w))
                words.append(w)
                
            # Else try lower value, else abandon this word
            elif w.lower() in self.word2id.keys():
                scores.append(self.cosine_similarity(word,w.lower()))
                words.append(w)
        
        # Get most similar words
        top_k_idx = np.argsort(scores)[-k:]
        
        # Extract the list of words and scores and reverse them
        top_k_words = np.array(words)[top_k_idx][::-1]
        top_k_scores = np.array(scores)[top_k_idx][::-1]

        return top_k_words, top_k_scores
        
        
        
    def closest_levenshtein(self, word, k=5):
        """
        Returns the k closet words of the training corpus 
        using levenshtein distance
        """
        # Compute distances between unknown and all words of the training corpus 
        lev_closer = []
        lev_distance = []
        for w in self.vocab:
            lev_closer.append(w)
            lev_distance.append(levenshtein_distance(word, w))
        
        # Keep closest words
        top_k_idx = np.argsort(lev_distance)[:k]
        
        lev_closer = np.array(lev_closer)[top_k_idx]
        lev_distance = np.array(lev_distance)[top_k_idx]

        return lev_closer, lev_distance
        
        
    def assign_token(self, unknown):
        """
        Assigns token to OoV word by looking at the most similar known words 
        (using embeddings and/or levenshtein) and their tag probabilities
        """
        
        # Checking if the OoV word is in the embeddings Polyglot vocab
        
        if unknown in self.vocab:
            # In that case:
            # Get 5 most similar words of training data according to cosine similarity
            most_sim, most_sim_scores = self.most_similar(unknown, k = 5)
            
            # Compute most probable tags as a weighted average
            proba_avg = np.average(self.lexicon[most_sim],axis=1,weights=most_sim_scores)
            
            # Find tag with highest proba
            probable_tag = self.lexicon.index[np.argmax(proba_avg)]
            
            return probable_tag
        
        
        # Else if the word is not in the embeddings vocab:
        # Look for words of similar spelling that are in the training data
        # or have an embedding
        else:
            lev_closer, lev_distance = self.closest_levenshtein(unknown, k = 5)
            
            most_sim = []
            weights = []
            
            
            for close_word, distance in zip(lev_closer, lev_distance):
                
                # For each of those closely spelled word, keep them if they are in the training data
                if close_word in self.words_training:
                    most_sim.append(close_word)
                    weights.append(1./distance)
                    
                # else find a similar one in the training data
                else:
                    sim_word, sim_score = self.most_similar(close_word, k = 1)
                    most_sim.append(sim_word[0])
                    weights.append(1./distance * sim_score[0])
            
            # Compute most probable tags as a weighted average
            proba_avg = np.average(self.lexicon[most_sim],axis=1,weights=weights)
            
            # Find tag with highest proba
            probable_tag = self.lexicon.index[np.argmax(proba_avg)]
            
            return probable_tag
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        