import time
from pathlib import Path
from argparse import ArgumentParser

from flask import Flask, request, jsonify

from predict import Predictor


def parse():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint_path', type=Path, help='Checkpoint path')
    parser.add_argument('--batch-size', type=int, default=16)
    return parser.parse_args()


args = parse()

app = Flask(__name__)
model = Predictor(args.config, args.checkpoint_path, args.batch_size)


@app.route("/")
def hello_world():
    return "Hello! I am model service"


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time_ns()
    if request.is_json:
        data = request.get_json()
        input_data = data['list_text']
        labels = model(input_data)
        print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
        return jsonify({"message": "predict successfully", "result": str(labels)}), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8080)
