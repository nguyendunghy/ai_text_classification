import argparse
import math

import more_itertools
from tqdm import tqdm

from src.model.data_augmentation import DataAugmentator
from src.model.my_datasets import HumanDataset, PromptDataset
from src.model.vllm_model import VLLMModel
from src.model.ollama import OllamaModel
from src.sql.database import engine, SessionLocal
from src.sql.models import Base, TextModel

Base.metadata.create_all(bind=engine)


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

        self.db = SessionLocal()

        print(f"DataGenerator initialized")

    def generate_ai_data(self, n_samples):
        print(f"Generating {n_samples} samples of AI data")

        processed = 0
        for i_model in tqdm(range(len(self.models)), desc=f"Generating AI data"):
            cnt_samples = int(n_samples * self.model_probs[i_model]) if i_model != len(
                self.models) - 1 else n_samples - processed
            self.models[i_model].init_model()
            model = self.models[i_model]
            model_name = self.model_names[i_model]

            print(f"Generating with {model_name} model and params {model.params}")
            for _ in tqdm(range(cnt_samples)):
                while True:
                    el = next(self.prompt_dataset)
                    el['text'] = model(el['prompt'], text_completion_mode=True)

                    el['model_name'] = model_name
                    el['model_params'] = str(model.params)

                    text, augs = self.augmentator(el['text'])
                    el['text'] = text
                    del el['topic']
                    el['augmentations'] = str(augs)

                    if len(el['text']) > self.min_text_length:
                        break

                row = TextModel(**el, label=True)
                self.db.add(row)
                self.db.commit()

            processed += cnt_samples

    def generate_human_data(self, n_samples):
        print(f"Generating {n_samples} samples of Human data")

        for _ in tqdm(range(n_samples), desc="Generating Humand Data"):
            while True:
                el = next(self.human_dataset)

                # text, augs = self.augmentator(el['text'])
                # el['text'] = text
                del el['topic']

                # el['augmentations'] = str(augs)

                if len(el['text']) > self.min_text_length:
                    break

            row = TextModel(**el, label=False)
            self.db.add(row)
            self.db.commit()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_size', type=int, help='Number of samples to generate')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    models = [
        OllamaModel(model_name='mistral:text'),
        OllamaModel(model_name='llama3:text'),
        OllamaModel(model_name='mixtral:text'),
        OllamaModel(model_name='gemma:7b'),
        OllamaModel(model_name='command-r'),
        OllamaModel(model_name='neural-chat'),
        OllamaModel(model_name='zephyr:7b-beta'),
        OllamaModel(model_name='openhermes'),
        OllamaModel(model_name='wizardcoder'),
        OllamaModel(model_name='starling-lm:7b-beta'),
        OllamaModel(model_name='yi:34b'),
        OllamaModel(model_name='openchat:7b'),
        OllamaModel(model_name='dolphin-mistral'),
        OllamaModel(model_name='solar'),
        OllamaModel(model_name='llama2:13b'),
    ]

    data_generator = DataGenerator(models, model_probs=None)
    # ai_data = data_generator.generate_ai_data(args.dataset_size)
    human_data = data_generator.generate_human_data(args.dataset_size)
