import os
import numpy as np
from musket_core import utils,preprocessing,context,model,datasets
from musket_core.context import get_current_project_data_path
from nltk.tokenize import casual_tokenize
from musket_core.datasets import DataSet
from musket_core import configloader
from musket_core import caches,metrics
from collections import Counter
import tqdm
import keras
from future.types import no
_loaded={}

def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    

def embeddings(EMBEDDING_FILE:str):
    path=context.get_current_project_data_path()
    emb=os.path.join(path,EMBEDDING_FILE)
    if EMBEDDING_FILE in _loaded:
        return _loaded[EMBEDDING_FILE]
    cache=path
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

@preprocessing.dataset_transformer
def pad_sequence_labeling(inp:datasets.DataSet,maxLen=-1)->datasets.DataSet:
    lm=inp
    if isinstance(inp, datasets.CompositeDataSet):
        lm=inp.components[0]
    no_token_class=lm.num2Class[lm.clazzColumn][2]   
    if "O" in lm.num2Class[lm.clazzColumn][0]:
        no_token_class=lm.num2Class[lm.clazzColumn][0]['O'] 
    def pad_sequence_label(x):
        tokenText=list(x.x)
        tokenClazz=list(x.y)
        
            
        while len(tokenClazz)<maxLen:
            tokenClazz.append(no_token_class)
            tokenText.append("eos")
        if len(tokenClazz)>maxLen:
            tokenClazz=tokenClazz[0:maxLen]
            tokenText=tokenText[0:maxLen]   
        return preprocessing.PreproccedPredictionItem(x.id,np.array(tokenText),np.array(tokenClazz),x)     
    rs= preprocessing.PreprocessedDataSet(inp,pad_sequence_label,True)
    return rs

@preprocessing.dataset_preprocessor
def tokens_to_case(inp):
    rr=[]
    for w in inp:
        rr.append([w.lower()==w,w.upper()==w,w.isdigit(),w.isalpha()])
    return np.array(rr)


@preprocessing.dataset_preprocessor
def lowercase(inp:str):
    if isinstance(inp,str):
        return inp.lower()
    return np.array([x.lower() for x in inp])



class CropFirst1(keras.layers.Layer):

    def build(self, input_shape):
        keras.layers.Layer.build(self, input_shape)
        
    def call(self, inp):
        return inp[:,0,:]    

    def compute_output_shape(self, input_shape):
        return (input_shape[1],input_shape[2])    

@model.block
def bert(inp):
    from musket_text.bert.load import load_google_bert
    cfg=inp[0].contribution
    path=cfg.path
    max_len=cfg.len
    g_bert, cfg = load_google_bert(get_current_project_data_path()+path + '/', max_len=max_len, use_attn_mask=False,customInputs=inp)
    outputs = g_bert.outputs
    return outputs[0]

def bertDeployHandler(p1,cfg,p2):
    try:
        contrib=p1.contribution
        if contrib.path[0]=='/':
            contrib.path=contrib.path[1:]
        nm=os.path.join(p2,"assets",contrib.path)
        import shutil
        shutil.copytree(get_current_project_data_path()+'/'+contrib.path + '/', nm)                
    except:
        import traceback
        traceback.print_exc()

@preprocessing.dataset_transformer
def text_to_bert_input(inp,path,max_len): 
    
    from musket_text.bert.bert_encoder import create_tokenizer
    from musket_text.bert.input_constructor import prepare_input
    bertTokenizer = create_tokenizer(get_current_project_data_path()+path)
    @preprocessing.deployHandler(bertDeployHandler)                   
    def transform2index(x):
        
    
        bInput = prepare_input(x, max_len, bertTokenizer, False)
        if bInput.attn_mask is not None:
            return [x[0] for x in [bInput.input_ids, bInput.input_type_ids, bInput.token_pos, bInput.attn_mask]]
        else:
            return [x[0] for x in [bInput.input_ids, bInput.input_type_ids, bInput.token_pos]]        
    rs= preprocessing.PreprocessedDataSet(inp,transform2index,False)
    rs.path=path
    rs.contribution=BertConfig(path,max_len)
    return rs

@model.block
def takeFirstToken(inp):
    return CropFirst1()(inp)

class BertConfig:
    def __init__(self,path,ln):
        self.path=path
        self.len=ln

@preprocessing.dataset_preprocessor
def tokenize(inp):
    try:
        return casual_tokenize(inp)
    except:
        print(inp)
        return []


class Vocabulary:
    def __init__(self,words):
        self.dict={}
        self.i2w={}
        num=0
        for c in words:        
            self.dict[c]=num
            self.i2w[num]=c
            num=num+1
        self.unknown=len(self.dict)
        
        
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
    words=counter.most_common(maxWords)
            
    return Vocabulary([str(x[0]).strip() for x in words])

_vocabs={}

def vocabularyDeployHandler(p1,cfg,p2):
    try:
        nm="data."+str(p1.maxWords)+".vocab"
        nm=os.path.join(p2,"assets",nm)        
        utils.save(nm,p1.vocabulary)
    except:
        import traceback
        traceback.print_exc()    


@preprocessing.dataset_transformer
def tokens_to_indexes(inp:DataSet,maxWords=-1,maxLen=-1)->DataSet:
    voc=caches.get_cache_dir()
    
    name=voc+caches.cache_name(inp)+"."+str(maxWords)+".vocab"
    # WE SHOULD USE TRAIN VOCABULARY IN ALL CASES
    rp=os.path.join(context.get_current_project_path(),"assets",caches.cache_name(inp)+"."+str(maxWords)+".vocab")
    vocabulary=None
    if os.path.exists(rp):  
        if name in _vocabs:
            vocabulary= _vocabs[name]
        else:    
            vocabulary=utils.load(rp)
            _vocabs[name]=vocabulary
    if vocabulary is None:        
        try:
            trainName=str(inp.root().cfg.dataset)
            
            curName=inp.root().name
            if trainName!=curName:
                name=utils.load(inp.root().cfg.path+".contribution")
                if isinstance(name , list):
                    name=name[0]              
        except:
            pass 
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
    @preprocessing.deployHandler(vocabularyDeployHandler)            
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
    rs.maxWords=maxWords
    rs.maxLen=maxLen
    rs.contribution=name
    return rs

def get_vocab(nm)->Vocabulary:
        
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
    
    embedding_matrix = None
    try:
        
        if context.isTrainMode():
            embs=embeddings(path)
            v=get_vocab(inp.contribution);
                
            for word, i in tqdm.tqdm(v.dict.items()):
                if embedding_matrix is None:
                    embedding_matrix=np.random.randn(len(v.dict)+1, len(embs[word]))
                    context.addTrainSetting((len(v.dict)+1, len(embs[word])))
                if word in embs:
                    embedding_matrix[i]=embs[word]
            
            return keras.layers.Embedding(len(v.dict)+1,embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)        
        else:
            s,z = context.popTrainSetting()
            return keras.layers.Embedding(s,z,trainable=False)(inp)    
    except:
        import traceback
        traceback.print_exc()
        return None     
    
from seqeval import metrics as sem   

class connll2003_entity_level_f1(metrics.ByOneMetric):
    def __init__(self):
        self.gt=[]
        self.pr=[]
        self.name="connll2003_entity_level_f1"
        pass
  
    def onItem(self,outputs,labels):
        vl=self.dataset
        if isinstance(vl, datasets.CompositeDataSet):
            vl=vl.components[0]
        labels=vl.decode(labels)
        gt=vl.decode(outputs,len(labels));
        self.pr=self.pr+labels
        self.gt=self.gt+gt
        pass
    
    def eval(self,predictions):
        self.dataset=predictions.root()
        return super().eval(predictions)
    
    def commit(self,dict):
        
        dict[self.name]=sem.f1_score(self.gt,self.pr)
        return dict
    
class connll2003_entity_level_precision(connll2003_entity_level_f1):
    def __init__(self):
        self.gt=[]
        self.pr=[]
        self.name="connll2003_entity_level_precision"
        pass
  
    
    
    def eval(self,predictions):
        self.dataset=predictions.root()
        return super().eval(predictions)
    
    def commit(self,dict):
        dict[self.name]=sem.precision_score(self.gt,self.pr)
        return dict
    
class connll2003_entity_level_recall(connll2003_entity_level_f1):
    def __init__(self):
        self.gt=[]
        self.pr=[]
        self.name="connll2003_entity_level_recall"
        pass
    
    def eval(self,predictions):
        self.dataset=predictions.root()
        return super().eval(predictions)
    
    def commit(self,dict):
        dict[self.name]=sem.recall_score(self.gt,self.pr)
        return dict        


configloader.load("layers").catalog[connll2003_entity_level_f1().name]=connll2003_entity_level_f1()    
configloader.load("layers").catalog[connll2003_entity_level_precision().name]=connll2003_entity_level_precision()
configloader.load("layers").catalog[connll2003_entity_level_recall().name]=connll2003_entity_level_recall()