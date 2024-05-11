import pandas
import pandas as pd
from tqdm import tqdm

from scripts.model.data_augmentation import DataAugmentator
from scripts.model.models import ValDataRow
from scripts.model.my_datasets import HumanDataset, PromptDataset
from scripts.model.text_completion import OllamaModel


class DataGenerator:
    def __init__(self, models: list, model_probs: list | None, min_text_length=250):
        print(f"DataGenerator initializing...")
        print(f"Models {models}")
        print(f"model_probs {model_probs}")

        self.min_text_length = min_text_length
        self.models = models
        self.model_names = [el.model_name for el in models]
        self.augmentator = DataAugmentator()

        if model_probs is None:
            self.model_probs = [1 / len(self.models) for i in range(len(self.models))]
        else:
            self.model_probs = model_probs
            assert sum(model_probs) == 1

        self.human_dataset = HumanDataset()
        self.prompt_dataset = PromptDataset()

        assert len(self.models) == len(self.model_names) == len(self.model_probs)
        assert len(self.models) != 0

        self._data_frame = pd.DataFrame(columns=list(ValDataRow.model_fields.keys()))

        print(f"DataGenerator initialized")

    def generate_ai_data(self, n_samples) -> list[ValDataRow]:
        print(f"Generating {n_samples} samples of AI data")

        res = []
        processed = 0
        for i in tqdm(range(len(self.models)), desc=f"Generating AI data"):
            cnt_samples = int(n_samples * self.model_probs[i]) if i != len(self.models) - 1 else n_samples - processed
            self.models[i].init_model()
            model = self.models[i]
            model_name = self.model_names[i]

            print(f"Generating with {model_name} model and params {model.params}")
            for j in tqdm(range(cnt_samples)):
                while True:
                    el = next(self.prompt_dataset)
                    el['text'] = model(el['prompt'], text_completion_mode=True)
                    el['model_name'] = model_name
                    el['model_params'] = model.params

                    text, augs = self.augmentator(el['text'])
                    el['text'] = text
                    el['augmentations'] = augs

                    if len(el['text']) > self.min_text_length:
                        break

                val_data_row = ValDataRow(**el, label=True)
                res.append(val_data_row)

            processed += cnt_samples
        return res

    def generate_human_data(self, n_samples) -> list[ValDataRow]:
        print(f"Generating {n_samples} samples of Human data")

        res = []
        for i in tqdm(range(n_samples), desc="Generating Humand Data"):
            while True:
                el = next(self.human_dataset)

                text, augs = self.augmentator(el['text'])
                el['text'] = text
                el['augmentations'] = augs

                if len(el['text']) > self.min_text_length:
                    break

            val_data_row = ValDataRow(**el, label=False)
            res.append(val_data_row)
        return res


if __name__ == "__main__":
    dataset_size = 5_000
    model_names = ['llama3', 'mistral', 'neural-chat', 'gemma:7b', 'solar']
    models = [OllamaModel(model_name) for model_name in model_names]
    data_generator = DataGenerator(models, [0.2, 0.2, 0.2, 0.2, 0.2])
    ai_data = data_generator.generate_ai_data(dataset_size)
    human_data = data_generator.generate_human_data(dataset_size)

    df = pd.DataFrame(columns=list(ValDataRow.model_fields.keys()))
    for ai_data_row in ai_data:
        df.loc[df.shape[0]] = ai_data_row.model_dump()
    for human_data_row in human_data:
        df.loc[df.shape[0]] = human_data_row.model_dump()
    df.to_csv('resources/data.csv', index=False)
