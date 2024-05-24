# Install
```shell
pip install -r requirements.txt
pip install ort-nightly-gpu==1.17.0.dev20231205004 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/
```

# Run
`python app.py $onnx_model --tokenizer='microsoft/deberta-v3-base' --batch-size=$batch-size`

onnx_model : Path to the ONNX model file.

tokenizer : str, The name of the tokenizer to use. The same with model architecture

--batch-size 16: Set batch size for prediction.

--port 8888: set port for api