import os
import numpy as np
from musket_core import utils,preprocessing,context,model
from nltk.tokenize import casual_tokenize
from musket_core.datasets import DataSet
from musket_core import caches
from collections import Counter
import tqdm
import keras
_loaded={}

def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    

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
        result= dict(get_coefs(*o.strip().split(" ")) for o in open(emb,encoding="utf8",errors="ignore") if len(o)>100)
    else:
        import gensim
        vectors=gensim.models.KeyedVectors.load_word2vec_format(emb, binary=True)
        result={}
        result.dict = vectors.vocab
        result.vectors = vectors.vectors
             
    _loaded[EMBEDDING_FILE]=result
    utils.save(cache+EMBEDDING_FILE+".embcache", result)
    return result


@preprocessing.dataset_preprocessor
def tokenize(inp):
    try:
        return casual_tokenize(inp)
    except:
        print(inp)
        return []


class Vocabulary:
    def __init__(self,voc:dict,i2w):
        self.dict=voc
        self.i2w=i2w
        self.unknown=len(voc)
        
        
def buildVocabulary(inp:DataSet,maxWords=None):
    counter=Counter()
    if maxWords==-1:
        maxWords=None
    for i in tqdm.tqdm(range(len(inp)),desc="Building vocabulary for:"+str(inp)):
        p=inp[i]        
        for c in p.x:
            counter[c]+=1
    word2Index={}  
    indexToWord={}      
    num=1
    for c in counter.most_common(maxWords):        
        word2Index[c[0]]=num
        indexToWord[num]=c[0]
        num=num+1        
    return Vocabulary(word2Index,indexToWord)


_vocabs={}


@preprocessing.dataset_transformer
def tokens_to_indexes(inp:DataSet,maxWords=-1,maxLen=-1)->DataSet:
    voc=caches.get_cache_dir()
    
    name=voc+caches.cache_name(inp)+"."+str(maxWords)+".vocab"
    if os.path.exists(name):
        if name in _vocabs:
            vocabulary= _vocabs[name]
        else:    
            vocabulary=utils.load(name)
            _vocabs[name]=vocabulary
    else:
        vocabulary=buildVocabulary(inp,maxWords)
        utils.save(name,vocabulary)
        _vocabs[name]=vocabulary    
    def transform2index(x):
        ml=maxLen
        if ml==-1:
            ml=len(x)
        res=np.zeros((ml,),dtype=np.int32)
        num=0
        for v in x:
            if v in vocabulary.dict:
                res[num]=(vocabulary.dict[v])
            else:
                res[num]=(vocabulary.unknown)
            num=num+1
            if num==ml:
                break            
        return res    
    rs= preprocessing.PreprocessedDataSet(inp,transform2index,False)
    rs.vocabulary=vocabulary
    rs.contribution=name
    return rs

def get_vocab(nm)->Vocabulary:
        
    print(nm)
    if nm in _vocabs:
        return _vocabs[nm]
    vocabulary=utils.load(nm)
    _vocabs[nm]=vocabulary
    return vocabulary
@preprocessing.dataset_transformer
def vectorize_indexes(inp,path,maxLen=-1):
    embs=embeddings(path)
    orig=inp
    while not hasattr(orig, "vocabulary"):
        orig=orig.parent
    voc=orig.vocabulary
    unknown=np.random.randn(300)    
    def index2Vector(inp):
        ml=maxLen
        if ml==-1:
            ml=len(inp)
        ln=min(ml,len(inp))
        result=np.zeros((ml,300),dtype=np.float32)        
        for i in range(ln):
            ind=inp[i]
            if ind==0:
                break
            if ind in voc.i2w:
                w=voc.i2w[ind]
                if w in embs:
                    result[i]=embs[w]
                    continue
            result[i]=unknown    
        return result                    
    rs= preprocessing.PreprocessedDataSet(inp,index2Vector,False)
    return rs    

@preprocessing.dataset_preprocessor
class vectorize:
    def __init__(self,path,maxLen=-1):
        self.embeddings=embeddings(path)
        self.maxLen=maxLen
        pass
    def __call__(self,inp):
        ml=self.maxLen
        if ml==-1:
            ml=len(inp)
        ln=min(ml,len(inp))
        result=np.zeros((ml,300),dtype=np.float32)        
        for i in range(ln):
            w=inp[i]
            if w in self.embeddings:
                result[i]=self.embeddings[w]
            else:    
                w=w.lower()
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

@preprocessing.dataset_preprocessor
def remove_random_words(inp,probability):
    rr=np.random.rand(len(inp))
    result=[]
    count=0
    for i in range(len(inp)):
        if rr[i]<probability:
            count=count+1
            continue
        result.append(inp[i])
    result=result+[0]*count
    return np.array(result)        


@preprocessing.dataset_preprocessor
def swap_random_words(inp,probability):
    rr=np.random.rand(len(inp))
    result=[]
    continueNext=False
    for i in range(len(inp)-1):
        if continueNext:
            continueNext=False
            continue
        if rr[i]<probability:
            result.append(inp[i+1])
            result.append(inp[i])
            continueNext=True
            continue
            
        result.append(inp[i])
    while len(result)<len(inp):
        result.append(0)    
    if len(result)!=len(inp):
        raise ValueError()      
    return np.array(result)

@preprocessing.dataset_preprocessor
def add_random_words(inp,probability):
    rr=np.random.rand(len(inp))
    result=[]
    for i in range(len(inp)):
        if rr[i]<probability:
            result.append(np.random.randint(1,2000))
            
            
        result.append(inp[i])
    if len(result)>len(inp):
        result=result[:len(inp)]      
    return np.array(result)        
 
 
    
@model.block    
def word_indexes_embedding(inp,path):
    embs=embeddings(path)
    v=get_vocab(inp.contribution);
    embedding_matrix = np.random.randn(len(v.dict)+1, 300)
    
    for word, i in tqdm.tqdm(v.dict.items()):
        if word in embs:
            embedding_matrix[i]=embs[word]
    return keras.layers.Embedding(len(v.dict)+1,300,weights=[embedding_matrix],trainable=False)(inp)    
        
    
#xz=string_to_chars(builtin_datasets.from_array(["Hello","Маруся"],[0,0]),maxLen=100,encoding="cp1251")
#print(xz[1].x)