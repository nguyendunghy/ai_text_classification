from torch import nn
import transformers


class DistilBert(nn.Module):
    def __init__(self, model_name, dropout, attention_dropout):
        super().__init__()

        config = transformers.DistilBertConfig(dropout=dropout, attention_dropout=attention_dropout)
        self.dbert = transformers.DistilBertModel.from_pretrained(model_name, config=config)

    def forward(self, x):
        x = self.dbert(input_ids=x)
        x = x["last_hidden_state"][:, 0, :]
        return x
