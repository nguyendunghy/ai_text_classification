import time
from argparse import ArgumentParser
from pathlib import Path

import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from flask import Flask, request, jsonify
from nltk import sent_tokenize

from predict import Predictor


def parse():
    parser = ArgumentParser()
    parser.add_argument('onnx_model', type=Path, default=Path('model.onnx'))
    parser.add_argument('--tokenizer', type=Path, default='microsoft/deberta-v3-base')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--port', type=int, default=8888)
    return parser.parse_args()


args = parse()

app = Flask(__name__)
model = Predictor(args.onnx_model, args.tokenizer, args.batch_size)


@app.route("/")
def hello_world():
    return "Hello! I am model service"


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time_ns()
    if request.is_json:
        data = request.get_json()
        input_data = data['list_text']

        texts = []
        for i, data in enumerate(input_data):
            sentences = sent_tokenize(data)
            if len(sentences) > 3:
                tails = sentences[1:]
                data = ' '.join(tails)
                texts.append(data)
            else:
                texts.append(data)

        labels = model(texts)
        print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
        response = {
            "message": "predict successfully",
            "result": labels
        }
        return jsonify(response), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=args.port)
