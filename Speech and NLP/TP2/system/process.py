import re


def remove_labels(labeled_sentences):
    """
    Removes all functional labels
    """
    raw_sentences = []
    for line in labeled_sentences:
        line = re.sub(r'\([a-zA-Z-_+]*\s','', line)
        line = re.sub(r'(\)|\n)','',line)
        line = line.replace('\n','')
        raw_sentences.append(line)
        
    return raw_sentences
    
    
    
def remove_nonterminal_labels(labeled_sentences):
    """
    Removes the functional labels of non-terminal tokens 
    """
    sentences = []
    non_terminal = "\(\w+(-\w+)"

    for labeled_sentence in labeled_sentences:
        sentence = labeled_sentence
 
        for w in re.findall(non_terminal,labeled_sentence):
            sentence = re.sub(w, "",sentence)
            sentence = sentence.replace('\n','')
            
        sentences.append(sentence)
            
    return sentences
    


