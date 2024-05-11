import torch
import numpy as np
import transformers


class DistilBertTokenizer:
    def __init__(self, model_name):
        self._tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)

    def tokenize(self, texts):
        input_ids = [self._tokenizer(text, padding='max_length', max_length=512, truncation=True)['input_ids']
                     for text in texts]
        input_ids = np.array(input_ids, dtype='int32')
        input_ids = torch.Tensor(input_ids).long()
        return input_ids
