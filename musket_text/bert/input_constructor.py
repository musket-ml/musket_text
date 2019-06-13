import numpy as np
import re

from musket_text.bert.extract_features import convert_examples_to_features, InputExample
from musket_text.bert.data.dataset import create_attention_mask, generate_pos_ids
from musket_text.bert import tokenization

class BertInput():
    def __init__(self,batch_size,input_ids,input_mask,input_type_ids,token_pos,tokens,attn_mask):
        self.batch_size = batch_size
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.token_pos = token_pos
        self.tokens = tokens
        self.attn_mask = attn_mask
        pass



def prepare_input(text, max_seq_len, tokenizer, use_attn_mask) -> BertInput:

    if not isinstance(text,list):
        text = [ text ]

    batch_size = len(text)
    examples = read_examples(text)
    features = convert_examples_to_features(examples, max_seq_len, tokenizer)
    token_ids = np.array([ x.input_ids for x in features ]).reshape(batch_size, -1)
    input_mask = np.array([ x.input_mask for x in features ]).reshape(batch_size, -1)
    positions = generate_pos_ids(batch_size, max_seq_len)
    input_type_ids = np.array([ x.input_type_ids for x in features ]).reshape(batch_size, -1)
    tokens = [ x.tokens for x in features ]
    attention_input_mask = None
    if use_attn_mask:
        attention_input_mask = create_attention_mask(input_mask, False, batch_size, None, True)
    input = BertInput(batch_size,token_ids, input_mask, input_type_ids, positions,tokens,attention_input_mask)
    return input


def read_examples(lst_strs):
    """Read a list of `InputExample`s from a list of strings."""
    unique_id = 0
    for ss in lst_strs:
        line = tokenization.convert_to_unicode(ss)
        if not line:
            continue
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
        unique_id += 1