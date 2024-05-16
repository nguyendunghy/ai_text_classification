import time

import requests as re

sequenses = [
    'From the first day of orientation, students took the lead in serving as tour guides and introducing speakers.',
    'Maximum entropy modeling of species geographic distributions.',
    'A. D. 13.',
    'Cal.',
    'Marine Ecology Progress Series 316, 285-310.',
    'Mapping world-wide distributions and marine mammal species using a relative environmental suitability (RES) model.',
    'Sir Stephen Fox, Knight, and of his wife Elizabeth, parents to son and son to parents most worthy....'
]


def post(ip):
    response = re.post(f'http://{ip}/predict', json={'list_text': sequenses})
    return response.text


if __name__ == '__main__':
    t1 = time.time()
    answer = post(ip='127.0.0.1:8080')
    t2 = time.time()
    print(answer)
    print(t2 - t1)
