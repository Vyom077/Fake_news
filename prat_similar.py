import numpy as np 
import pandas as pd
from transformers import AutoTokenizer, AutoModel # for tokenizing and word embeddings
import torch # for tensor processing

# for data preprocessing
import nltk
from nltk.tokenize import word_tokenize
import string
import re

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class Preprocesser():
    def __init__(self):
        self.q1 = None
        self.q2 = None
        return None

    def preprocess(self, doc):
        doc = doc.lower()
        doc = re.sub('[^a-zA-Z1-9]', ' ', doc)
        return doc

    def process_docs(self, documents):
        new_docs = [self.preprocess(doc) for doc in documents]
        return new_docs
    
    # The 2 main functions
    def fit(self, X, y=None):
        self.q1 = X.iloc[:, 0]
        self.q2 = X.iloc[:, 1]
        return self
    
    def transform(self, X, y=None):
        X_new = pd.DataFrame(np.zeros(shape=X.shape))
        
        newq1 = self.process_docs(self.q1)
        newq2 = self.process_docs(self.q2)
        
        X_new.iloc[:, 0] = newq1
        X_new.iloc[:, 1] = newq2
        return X_new
    

class MiniLMModel():
    def __init__(self):
        self.q1 = None
        self.q2 = None
        return None

    def get_embeddings(self, docs):
        model_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        tokens = {
            'input_ids' : [],
            'attention_mask' : []
        }

        for doc in docs:
            new_tokens = tokenizer.encode_plus(doc, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens['attention_mask'].append(new_tokens['attention_mask'][0])

        tokens['input_ids'] = torch.stack(tokens['input_ids'])
        tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

        outputs = model(**tokens)

        embeddings = outputs['last_hidden_state']
        attention = tokens['attention_mask']
        mask = attention.unsqueeze(-1).expand(embeddings.shape).float()

        # important line, helps to get only the embeddings for the useful words
        mask_embeddings = embeddings * mask

        summed = torch.sum(mask_embeddings, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts

        # using this line since i want to get a numpy array for calculations, not a tensor
        mean_pooled = mean_pooled.detach().numpy()
        return mean_pooled

    def similar(self, docs1, docs2):
        q1_emb = self.get_embeddings(list(docs1))
        q2_emb = self.get_embeddings(list(docs2))
        
        sim = np.array([cosine_similarity([q1_emb[i], q2_emb[i]])[0, 1] for i in range(len(q1_emb))])
        return sim
    
    def fit(self, X, y=None):
        self.q1 = X.iloc[:, 0]
        self.q2 = X.iloc[:, 1]
        return self
    
    def transform(self, X, y=None):
        X_new = pd.DataFrame(np.zeros(shape=X.shape[0]), columns=['semnatic_similarity'])
        X_new.iloc[:, 0] = self.similar(self.q1, self.q2)
        
        X_cpy = X.copy(deep=True)
        X_cpy.reset_index(inplace=True)
        
        X_new2 = pd.concat([X_cpy, X_new], axis=1)
        X_new2.drop(X_new2.columns[0], axis=1, inplace=True)
        return X_new2
    

class JacIdx():
    def __init__(self):
        self.q1 = None
        self.q2 = None
        return None
    
    def fit(self, X, y=None):
        self.q1 = X.iloc[:, 0]
        self.q2 = X.iloc[:, 1]
        return self
    
    def transform(self, X, y=None):
        jacc = []
        docs1, docs2 = self.q1, self.q2
        for i in range(len(docs1)):
        #     print(ok3.iloc[i, 0], ok3.iloc[i, 1])
            s1 = set(word_tokenize(docs1.iloc[i]))
            s2 = set(word_tokenize(docs2.iloc[i]))
            jacsimilar = len(s1.intersection(s2)) / len(s1.union(s2))
            jacc.append(jacsimilar)
        jacc = pd.DataFrame(jacc, columns=['jaccard_similarity'])
        
        X_new = pd.concat([X, jacc], axis=1)
        return X_new
    

class ExtraProcessor:
    def __init__(self):
        self.data = None
        return None
    
    def fit(self, X, y=None):
        self.data = X
        return self
    
    def transform(self, X, y=None):
        X_new = self.data.iloc[:, 2:]
        X_new.columns = X_new.columns.astype(str)
        return X_new
    

class BOWAdder2:
    def __init__(self, vocab_):
        self.q1 = None
        self.q2 = None
        self.vocab_ = vocab_
        return None
    
    def fit(self, X, y=None):
        self.q1 = X.iloc[:, 0]
        self.q2 = X.iloc[:, 1]
        return self
    
    def transform(self, X, y=None):
        cv = CountVectorizer(vocabulary=self.vocab_, max_features=10000, ngram_range=(1,1))
        
        questions = list(self.q1) + list(self.q2)
        q1_vecs, q2_vecs = np.vsplit(cv.fit_transform(questions).toarray(), 2)
        q1_vecs_df, q2_vecs_df = pd.DataFrame(q1_vecs), pd.DataFrame(q2_vecs)
        
        self.vocab_ = cv.vocabulary_
        
        X_cpy = X.copy(deep=True)
        X_cpy.reset_index(inplace=True)
        
        X_new = pd.concat([X_cpy, q1_vecs_df, q2_vecs_df], axis=1)
        X_new.drop(X_new.columns[0], axis=1, inplace=True)
        return X_new
    
