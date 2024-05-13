import time

import numpy as np
from langchain_community.llms import VLLM

from model.text_postprocessing import TextCleaner


class VLLMModel:
    def __init__(self, model_name, num_predict=1000, tensor_parallel_size=1):
        """
        available models you can find on https://github.com/ollama/ollama
        before running modeling <model_name> install ollama and run 'ollama pull <model_name>'
        """
        print(f'Initializing vllmModel {model_name}')
        if num_predict > 1000:
            raise Exception(
                "You're trying to set num_predict to more than 1000, it can lead to context overloading and Ollama hanging")

        self.model_name = model_name
        self.num_predict = num_predict
        self.tensor_parallel_size = tensor_parallel_size
        self.model = None
        self.params = {}
        # self.init_model()

        self.text_cleaner = TextCleaner()

    def init_model(self):
        sampling_temperature = np.clip(np.random.normal(loc=1, scale=0.2), a_min=0, a_max=2)
        # Centered around 1 because that's what's hardest for downstream classification models.
        frequency_penalty = np.random.uniform(low=0.9, high=1.5)
        top_k = int(np.random.choice([-1, 20, 40]))
        top_k = top_k if top_k != -1 else None
        top_p = np.random.uniform(low=0.5, high=1)

        self.model = VLLM(model=self.model_name,
                          timeout=200,
                          trust_remote_code=True,
                          # num_thread=1,
                          tensor_parallel_size=self.tensor_parallel_size,
                          num_predict=self.num_predict,
                          temperature=sampling_temperature,
                          repeat_penalty=frequency_penalty,
                          vllm_kwargs={
                              "quantization": "awq",
                          },
                          dtype="float16",
                          # top_p=top_p,
                          # top_k=top_k
                          )
        self.params = {'top_k': top_k, 'top_p': top_p, 'temperature': sampling_temperature,
                       'repeat_penalty': frequency_penalty}

    def __call__(self, prompts: list[str], text_completion_mode=False) -> str | None:
        # while True:
        try:
            # if text_completion_mode:
            # if ':text' not in self.model_name:
            #     system_message = "You're a text completion modeling, just complete text that user sended you"  # . Return text without any supportive - we write add your result right after the user text
            #     text = self.model.invoke([{'role': 'system', 'content': system_message},
            #                               {'role': 'user', 'content': prompt}])
            # else:
            # text = await self.model.ainvoke(prompt)
            # else:
            # assert ':text' not in self.model_name
            texts = self.model.batch(prompts)
            texts = [self.text_cleaner.clean_text(text) for text in texts]
            return texts
        except Exception as e:
            print(e)
            # print("Couldn't get response from Ollama, probably it's restarting now: {}".format(e))
            # time.sleep(1)

    def __repr__(self) -> str:
        return f"{self.model_name}"


if __name__ == '__main__':
    print("started")
    model = VLLMModel('TheBloke/Llama-2-7b-Chat-AWQ')
    # model = VLLMModel('TheBloke/Mistral-7B-Instruct-v0.1-AWQ')
    model.init_model()
    print("invoking")

    t1 = time.time()
    for i in range(16):
        model(prompts=['What is it?'])
    t2 = time.time()
    for i in range(4):
        model(prompts=['What is it?'] * 4)
    t3 = time.time()
    model(prompts=['What is it?'] * 16)
    t4 = time.time()

    print(t2 - t1, t3 - t2, t4 - t3)
    print("finished")
    # print(model.model)
