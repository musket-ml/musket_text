import os
import numpy as np
from musket_core import utils,preprocessing,context
from nltk.tokenize import casual_tokenize

_loaded={}

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')



def embeddings(EMBEDDING_FILE:str):
    path=context.get_current_project_path()
    emb=path+"/data/"+EMBEDDING_FILE
    if EMBEDDING_FILE in _loaded:
        return _loaded[EMBEDDING_FILE]
    cache=path+"/data/"
    utils.ensure(cache)
    if os.path.exists(cache+EMBEDDING_FILE+".embcache"):
        result=utils.load(cache+EMBEDDING_FILE+".embcache")
        _loaded[EMBEDDING_FILE]=result
        return result
        
    if not EMBEDDING_FILE.endswith(".bin"):
        result= dict(get_coefs(*o.split(" ")) for o in open(emb,encoding="utf8",errors="ignore") if len(o)>100)
    else:
        import gensim
        vectors=gensim.models.KeyedVectors.load_word2vec_format(emb, binary=True)
        result.dict = vectors.vocab
        result.vectors = vectors.vectors
             
    _loaded[EMBEDDING_FILE]=result
    utils.save(cache+EMBEDDING_FILE+".embcache", result)
    return result


@preprocessing.dataset_preprocessor
def tokenize(inp):
    return casual_tokenize(inp)

@preprocessing.dataset_preprocessor
class vectorize:
    def __init__(self,path,maxLen):
        self.embeddings=embeddings(path)
        self.maxLen=maxLen
        pass
    def __call__(self,inp):
        result=np.zeros((self.maxLen,300),dtype=np.float32)
        
        for i in range(self.maxLen):
            if i<len(inp):
                w=inp[i].lower()
                if w in self.embeddings:
                    result[i]=self.embeddings[w]
        return result

@preprocessing.dataset_preprocessor
class string_to_chars:
    
    def __init__(self,maxLen,encoding="utf8",errors='strict'):
        self.maxLen=maxLen
        self.encoding=encoding
        self.errors=errors
        
    def __call__(self,inp:str):
        vl=np.frombuffer(inp.encode(self.encoding, errors=self.errors),dtype=np.uint8)
        if vl.shape[0]<self.maxLen:
            r= np.pad(vl, (0,self.maxLen-vl.shape[0]),mode="constant")
            return r
        return vl[:self.maxLen]
    
    
#xz=string_to_chars(builtin_datasets.from_array(["Hello","Маруся"],[0,0]),maxLen=100,encoding="cp1251")
#print(xz[1].x)