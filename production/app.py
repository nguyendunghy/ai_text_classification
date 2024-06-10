import time
from pathlib import Path
from argparse import ArgumentParser

from flask import Flask, request, jsonify

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
        print(f'length of input: {len(input_data)}')
        labels = model(input_data)
        print(f'labels = {labels}')
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
