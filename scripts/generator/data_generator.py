import argparse
import math
import time
from pathlib import Path

import more_itertools
import pandas as pd
from tqdm import tqdm

from model.data_augmentation import DataAugmentator
from model.models import ValDataRow
from model.my_datasets import HumanDataset, PromptDataset
from scripts.generator.vllm_model import VLLMModel


class DataGenerator:
    def __init__(self, models: list, model_probs: list | None, batch_size=16, min_text_length=250):
        print(f"DataGenerator initializing...")
        print(f"Models {models}")
        print(f"model_probs {model_probs}")

        self.min_text_length = min_text_length
        self.batch_size = batch_size
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
        for i_model in tqdm(range(len(self.models)), desc=f"Generating AI data"):
            cnt_samples = int(n_samples * self.model_probs[i_model]) if i_model != len(
                self.models) - 1 else n_samples - processed
            self.models[i_model].init_model()
            model = self.models[i_model]
            model_name = self.model_names[i_model]

            print(f"Generating with {model_name} model and params {model.params}")
            for batch in tqdm(more_itertools.chunked(range(cnt_samples), self.batch_size),
                              total=math.ceil(cnt_samples / self.batch_size)):
                els = [next(self.prompt_dataset) for _ in range(len(batch))]
                prompts = [el['prompt'] for el in els]
                texts = model(prompts, text_completion_mode=True)

                for i in range(len(batch)):
                    el = els[i]
                    el['text'] = texts[i]
                    el['model_name'] = model_name
                    el['model_params'] = model.params

                    text, augs = self.augmentator(el['text'])
                    el['text'] = text
                    el['augmentations'] = augs

                for el in els:
                    if len(el['text']) > self.min_text_length:
                        val_data_row = ValDataRow(**el, label=True)
                        res.append(val_data_row)
            model.shotdown()
            print("sleeep")
            time.sleep(5)

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


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_size', type=int, help='Number of samples to generate')
    parser.add_argument('batch_size', type=int, help='Batch size', default=16)
    parser.add_argument("--output_csv", type=Path, default="resources/data.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    models = [
        VLLMModel('TheBloke/Llama-2-7b-Chat-AWQ', tensor_parallel_size=1, mode='completion'),
    ]

    data_generator = DataGenerator(models, model_probs=None, batch_size=args.batch_size)
    ai_data = data_generator.generate_ai_data(args.dataset_size)
    human_data = data_generator.generate_human_data(args.dataset_size)

    df = pd.DataFrame(columns=list(ValDataRow.model_fields.keys()))
    for ai_data_row in tqdm(ai_data, desc="AI Data"):
        df.loc[df.shape[0]] = ai_data_row.model_dump()
    for human_data_row in tqdm(human_data, desc="Human Data"):
        df.loc[df.shape[0]] = human_data_row.model_dump()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(args.output_csv), index=False)
