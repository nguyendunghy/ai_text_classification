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
            for batch in tqdm(more_itertools.chunked(range(cnt_samples), self.batch_size),
                              total=math.ceil(cnt_samples / self.batch_size)):
                els = [next(self.prompt_dataset) for _ in range(len(batch))]
                prompts = [el['prompt'] for el in els]
                texts = model(prompts, text_completion_mode=True)

                for i in range(len(batch)):
                    el = els[i]
                    el['text'] = texts[i]
                    el['model_name'] = model_name
                    el['model_params'] = str(model.params)

                    text, augs = self.augmentator(el['text'])
                    el['text'] = text
                    del el['topic']
                    el['augmentations'] = str(augs)

                for el in els:
                    if el and len(el['text']) > self.min_text_length:
                        row = TextModel(**el, label=True)
                        self.db.add(row)
                        self.db.commit()
            model.shotdown()

            processed += cnt_samples

    def generate_human_data(self, n_samples):
        print(f"Generating {n_samples} samples of Human data")

        for _ in tqdm(range(n_samples), desc="Generating Humand Data"):
            while True:
                el = next(self.human_dataset)

                text, augs = self.augmentator(el['text'])
                el['text'] = text
                del el['topic']

                el['augmentations'] = str(augs)

                if len(el['text']) > self.min_text_length:
                    break

            row = TextModel(**el, label=False)
            self.db.add(row)
            self.db.commit()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_size', type=int, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
    parser.add_argument('--gpus', type=int, help='Batch size', default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    models = [
        VLLMModel('casperhansen/llama-3-8b-instruct-awq', tensor_parallel_size=args.gpus, mode='completion',
                  vllm_kwargs=dict(
                      quantization='awq'
                  )),
        VLLMModel('TheBloke/Mistral-7B-Instruct-v0.2-AWQ', tensor_parallel_size=args.gpus, mode='completion',
                  vllm_kwargs=dict(
                      quantization='awq'
                  )),
        VLLMModel('casperhansen/gemma-7b-it-awq', tensor_parallel_size=args.gpus, mode='completion',
                  vllm_kwargs=dict(
                      quantization='awq'
                  )),
        VLLMModel('TheBloke/neural-chat-7B-v3-3-AWQ', tensor_parallel_size=args.gpus, mode='completion',
                  vllm_kwargs=dict(
                      quantization='awq'
                  )),
        VLLMModel('TheBloke/zephyr-7B-beta-AWQ', tensor_parallel_size=args.gpus, mode='completion',
                  vllm_kwargs=dict(
                      quantization='awq'
                  )),
        VLLMModel('TheBloke/OpenHermes-2.5-Mistral-7B-16k-AWQ', tensor_parallel_size=args.gpus, mode='completion',
                  vllm_kwargs=dict(
                      quantization='awq'
                  )),
        # VLLMModel('TheBloke/WizardCoder-33B-V1.1-AWQ', tensor_parallel_size=args.gpus, mode='completion',
        #           vllm_kwargs=dict(
        #               max_model_len=37200,
        #               quantization='awq'
        #           )),
        VLLMModel('TheBloke/Starling-LM-7B-alpha-AWQ', tensor_parallel_size=args.gpus, mode='completion',
                  vllm_kwargs=dict(
                      quantization='awq'
                  )),
        # VLLMModel('TheBloke/Yi-34B-Chat-AWQ', tensor_parallel_size=args.gpus, mode='completion',
        #           vllm_kwargs=dict(
        #               tensor_parallel_size=args.gpus,
        #               # max_model_len=37200,
        #               quantization='awq'
        #           )),
        VLLMModel('TheBloke/openchat_3.5-AWQ', tensor_parallel_size=args.gpus, mode='completion',
                  vllm_kwargs=dict(
                      quantization='awq'
                  )),
        VLLMModel('TheBloke/dolphin-2.6-mistral-7B-AWQ', tensor_parallel_size=args.gpus, mode='completion',
                  vllm_kwargs=dict(
                      quantization='awq'
                  )),
        VLLMModel('TheBloke/SOLAR-10.7B-Instruct-v1.0-AWQ', tensor_parallel_size=args.gpus, mode='completion',
                  vllm_kwargs=dict(
                      quantization='awq'
                  )),
        VLLMModel('TheBloke/Llama-2-13B-chat-AWQ', tensor_parallel_size=args.gpus, mode='completion',
                  vllm_kwargs=dict(
                      quantization='awq'
                  )),
    ]

    data_generator = DataGenerator(models, model_probs=None, batch_size=args.batch_size)
    # ai_data = data_generator.generate_ai_data(args.dataset_size)
    human_data = data_generator.generate_human_data(args.dataset_size)


