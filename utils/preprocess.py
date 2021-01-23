''' Handling the data io '''

import zipfile
from io import open
import os
import unicodedata
import string
import re
import random
import ast


# Read data
def extractZipFiles(zip_file, extract_to):
    ''' Extract from zip '''
    with zipfile.ZipFile(zip_file, 'r')as zipped_ref:
        zipped_ref.extractall(extract_to)
    print('done')
extractZipFiles(corpus_path, data_path)


def getLinesFromCorpus(path):
    """
    """
    lines = {}
    with open(path, 'r') as f:
        for line in f:
            columns = line.split(seperator)
            lines[columns[0]] = columns[-1]
    
    return lines

def loadConversations(convo_path, lines):
    conversations = []
    
    with open(convo_path, 'r') as f:
        for line in f:
            columns = line.split(seperator)
            convs = ast.literal_eval(columns[-1])
            for i, field in enumerate(convs[:-1]):
                context = lines[convs[i]].strip()
                response = lines[convs[i+1]].strip()
                conversations.append((context, response))
    return conversations



# Formating
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'MN')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s



def trim(list_of_pairs, max_length=MAX_LENGTH):
        
        pairs = []
        for pair in list_of_pairs:
            
            context = pair[0]
            response = pair[1]
            
            clean_context_tokens = normalizeString(context).split(' ')
            clean_response_tokens = normalizeString(response).split(' ')
            
            if len(clean_context_tokens) > max_length or len(clean_response_tokens) \
                                                                > max_length:
                continue
                
            pairs.append((clean_context_tokens, clean_response_tokens))
            
        return pairs


class WordVoc:
    
    def __init__(self, name):
        
        self.name = name
        
        
        self.PAD_token = 0 # for padding short sentences 
        self.SOS_token = 1 # Start-of-sentence token
        self.EOS_token = 2 # End-of-sentence token
        
        self.word2index = {}
        self.index2word = {self.PAD_token:'PAD', self.SOS_token:'SOS',
                          self.EOS_token: 'EOS'}
        self.word2count = {}
        self.num_words = 3 # Count, SOS, EOS, PAD
        
        
        
    def addSentence(self, sentence):
        for word in sentence:
            self.__addWord(word)
            
            
    def __addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
            
   