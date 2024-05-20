import time
import json
from pathlib import Path
from argparse import ArgumentParser

import tqdm
import more_itertools
import numpy as np
import onnxruntime as ort
import transformers

print(ort.get_device())


def parse():
    parser = ArgumentParser()
    parser.add_argument('onnx_model', type=Path, default=Path('model.onnx'))
    parser.add_argument('--tokenizer', type=Path, default='microsoft/deberta-v3-base')
    parser.add_argument('--batch-size', type=int, default=16)
    return parser.parse_args()


class Predictor:
    def __init__(self, onnx_model: Path, tokenizer, batch_size: int):
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self._model = ort.InferenceSession(str(onnx_model), providers=['CUDAExecutionProvider'])
        self._batch_size = batch_size

    def get_output_names(self):
        return [model_output.name for model_output in self._model.get_outputs()]

    def get_input_names(self):
        return [model_input.name for model_input in self._model.get_inputs()]

    def tokenize(self, text: str):
        tokens = self._tokenizer(text, padding='max_length', max_length=512, truncation=True)
        return dict(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask']
        )

    def forward(self, texts):
        tokens = [self.tokenize(text) for text in texts]
        input_ids = np.array([x['input_ids'] for x in tokens], dtype=np.int32)
        attention_mask = np.array([x['attention_mask'] for x in tokens], dtype=np.int32)
        ort_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
        ort_outs = self._model.run(self.get_output_names(), ort_inputs)
        return ort_outs[0]

    def __call__(self, texts):
        confs = []
        for batch in more_itertools.chunked(texts, self._batch_size):
            outputs = self.forward(batch)
            confs.extend(outputs)
        labels = [bool(conf > 0.5) for conf in confs]
        return labels


def load_test_data():
    with open('resources/test_data_2.json', 'r') as f:
        data = json.load(f)
        texts = data['data']
    return texts


def main(args):
    predictor = Predictor(args.onnx_model, args.tokenizer, args.batch_size)

    texts = load_test_data()
    t1 = time.time()
    labels = predictor(texts)
    print(labels)
    t2 = time.time()
    print(f"Time: {t2 - t1:.2f} s")


if __name__ == '__main__':
    args = parse()
    main(args)
