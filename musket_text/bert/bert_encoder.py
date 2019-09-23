import os
import numpy as np

from musket_text.bert.tokenization import FullTokenizer, convert_to_unicode,whitespace_tokenize
from musket_text.bert.load import load_google_bert
from musket_text.bert.input_constructor import prepare_input

__models = [
    "uncased_L-12_H-768_A-12",
    "uncased_L-24_H-1024_A-16",
    "multilingual_L-12_H-768_A-12",
    "cased_L-12_H-768_A-12",
    "cased_L-24_H-1024_A-16",
    "multi_cased_L-12_H-768_A-12" ]

__default_encoders:dict = {}

class BertEncodingResult():
    def __init__(self, batch_size, embeddings, attention_scores, tokens):
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.attention_scores = attention_scores
        self.tokens = tokens
        self.start_token_offset = 1


class KerasBertEncoder():

    def __init__(self, bertDir:str, bertModel, max_seq_len, use_attn_mask:bool=True, verbose=False, __models =[]):

        '''
        :param bertDir:
        :param bertModel: one of
            'uncased_L-12_H-768_A-12'
            'uncased_L-24_H-1024_A-16'
            'multilingual_L-12_H-768_A-12'
            'cased_L-12_H-768_A-12'
            'cased_L-24_H-1024_A-16'
            'multi_cased_L-12_H-768_A-12'
            or integer 0 ... 5 which is treated as index of one of these strings
        :param max_seq_len: length of output vector sequence
        :param use_attn_mask: whether to use attention mask, defaults True
        :param verbose: whether to print summary etc
        '''

        self.verbose = verbose

        if not bertDir.endswith("/"):
            bertDir += "/"

        if isinstance(bertModel,str):
            if len([x for x in __models if x == bertModel]) == 0:
                raise Exception(f"'bertModel' param should be integer from range(0,{len(__models)}) or string from[{__models.mkstring(', ')}]")
        elif isinstance(bertModel,int):
            bertModel = __models[bertModel]


        self.BERT_MODEL = bertDir + bertModel
        self.BERT_PRETRAINED_DIR = os.path.realpath(self.BERT_MODEL)
        self.tokenizer = create_tokenizer(self.BERT_PRETRAINED_DIR)

        self.max_seq_len = max_seq_len

        self.use_attn_mask = use_attn_mask
        self.g_bert, self.cfg = load_google_bert(base_location=self.BERT_PRETRAINED_DIR+'/', max_len = self.max_seq_len, use_attn_mask=self.use_attn_mask)
        if self.verbose:
            self.g_bert.summary()
        self.g_bert.compile('adam', 'mse')

    def encode(self, text):

        """
            Encode text string to a sequence of BERT vectors
            text: str or list[str]

            :return BERT encoding result -- embeddings, attention scores and tokens in a single structure
        """

        input = prepare_input(text, self.max_seq_len, self.tokenizer, self.use_attn_mask)
        predicted = self.g_bert.predict([input.input_ids, input.input_type_ids, input.token_pos, input.attn_mask],
                                        batch_size=input.batch_size, verbose=1)

        embeddings = np.array(predicted[0], dtype=np.float)
        scores = np.array(predicted[1:], dtype=np.float)
        result = BertEncodingResult(input.batch_size, embeddings, scores, input.tokens)
        return result


def get_default_encoder(bertDir,bertModel, max_seq_len, use_attn_mask:bool=True)->KerasBertEncoder:
    global __default_encoders
    global __models
    key = f'{bertDir}/{bertModel}/{str(max_seq_len)}/{str(use_attn_mask)}'
    if not key in __default_encoders:
        kbe = KerasBertEncoder(bertDir,bertModel,max_seq_len,use_attn_mask,True,__models)
        __default_encoders[key] = kbe
    return __default_encoders[key]

class WordpieceTokenizer1(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word
    self.unknowns = None

  def setUnknownsStorage(self,map:dict):
      self.unknowns = map

  def tokenize(self, text):
    """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """

    text = convert_to_unicode(text)

    output_tokens = []
    if (text in self.vocab):
      output_tokens.append(text)
    else:
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
        
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
        
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
    return output_tokens


def create_tokenizer(bertDir: str, unknownsMap = None):
    if not bertDir.endswith("/"):
        bertDir += "/"

    tokenizer = FullTokenizer(bertDir + "vocab.txt")
    wpt1 = WordpieceTokenizer1(tokenizer.vocab)
    wpt1.setUnknownsStorage(unknownsMap)
    tokenizer.wordpiece_tokenizer = wpt1

    return tokenizer
