from torch import nn
import transformers


class AutoModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        config = transformers.AutoConfig.from_pretrained(model_name)
        self.model = transformers.AutoModel.from_pretrained(model_name, config=config)

    def forward(self, x):
        x = self.model(x['input_ids'], attention_mask=x['attention_mask'])
        x = x["last_hidden_state"][:, 0, :]
        return x
