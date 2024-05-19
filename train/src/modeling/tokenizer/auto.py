import transformers


class AutoTokenizer:
    def __init__(self, model_name, max_seq_len):
        self._max_seq_len = max_seq_len
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def __call__(self, text: str):
        batch_encoding = self._tokenizer(text, padding='max_length', max_length=self._max_seq_len,
                                         truncation=True, return_tensors='pt')
        return dict(
            input_ids=batch_encoding['input_ids'].squeeze(),
            attention_mask=batch_encoding['attention_mask'].squeeze(),
        )
