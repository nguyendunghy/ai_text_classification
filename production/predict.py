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
    parser.add_argument('--batch-size', type=int, default=16)
    return parser.parse_args()


class Predictor:
    def __init__(self, onnx_model: Path):
        self._tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self._model = ort.InferenceSession(str(onnx_model), providers=['CUDAExecutionProvider'])

    def get_output_names(self):
        return [model_output.name for model_output in self._model.get_outputs()]

    def get_input_names(self):
        return [model_input.name for model_input in self._model.get_inputs()]

    def forward(self, texts):
        input_ids = [self._tokenizer(text, padding='max_length', max_length=512, truncation=True)['input_ids']
                     for text in texts]
        input_ids = np.array(input_ids, dtype='int32')
        ort_inputs = {self.get_input_names()[0]: input_ids}
        ort_outs = self._model.run(self.get_output_names(), ort_inputs)
        return ort_outs[0]


def load_test_data():
    with open('resources/test_data_2.json', 'r') as f:
        data = json.load(f)
        texts = data['data']
    return texts


def main(args):
    predictor = Predictor(args.onnx_model)

    texts = load_test_data()
    t1 = time.time()
    for batch in tqdm.tqdm(more_itertools.chunked(texts, 16)):
        confs = predictor.forward(batch)
        labels = confs > 0.5
        print(labels)
    t2 = time.time()
    print(f"Time: {t2 - t1:.2f} s")


if __name__ == '__main__':
    args = parse()
    main(args)
