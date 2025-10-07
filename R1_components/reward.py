import html
import math
import re
from collections import Counter
import random
from typing import List
from urllib import parse

import requests
from transformers import AutoTokenizer

def language_detection(origin_string):
    url = "https://www.google.com/async/translate"
    sl = 'auto'
    tl = 'en'
    origin_string = parse.quote(origin_string)

    payload = f"async=translate,sl:{sl},tl:{tl},st:{origin_string},id:1672{''.join([str(random.randint(0,9)) for i in range(4)])}60,qc:true,ac:true,_id:tw-async-translate,_pms:s,_fmt:pc"
    # payload = parse.quote(payload)
    headers = {
    'sec-ch-ua': '"Not?A_Brand";v="8", "Chromium";v="108", "Google Chrome";v="108"',
    'DNT': '1',
    'sec-ch-ua-mobile': '?0',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'sec-ch-ua-arch': '"x86"',
    'sec-ch-ua-full-version': '"108.0.5359.125"',
    'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
    'sec-ch-ua-platform-version': '"10.0.0"',
    'sec-ch-ua-full-version-list': '"Not?A_Brand";v="8.0.0.0", "Chromium";v="108.0.5359.125", "Google Chrome";v="108.0.5359.125"',
    'sec-ch-ua-bitness': '"64"',
    'sec-ch-ua-model': '',
    'sec-ch-ua-wow64': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'Accept': '*/*',
    'X-Client-Data': 'CKW1yQEIhbbJAQiktskBCMS2yQEIqZ3KAQjb08oBCLD+ygEIlaHLAQjv8swBCN75zAEI5PrMAQjxgM0BCLKCzQEI7ILNAQjIhM0BCO+EzQEIt4XNAQ==',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Secetch-Dest': 'empty',
    'host': 'www.google.com',
    'Cookie': '1P_JAR=2022-12-26-12; NID=511=eVLI1bG9nhyOZtqU14JBHm5Be00epdxfR4XmfQeehYyIkzgpXi6dbpNY75ZMVyS7aOjoM2oZ5WdoR8eNq6wi1-e_J0NeoyI0dtsHW-_8Ik4PGrqvuGHdcvVC03zTOEK2TY1FZL85Wimo_ZPIE3hGIrmGPSiel6-rRRW9lD30UPs'
    }

    response = requests.request("POST", url, headers=headers, data=payload).text
    translated = re.findall(r'<span[^>]*id="tw-answ-detected-sl">(.*?)</span>', response)[0]
    translated = html.unescape(html.unescape(translated))
    return translated


class Rewards:
    def __init__(self, tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")):
        self.config = {
            'best_cot_length': 256,
        }
        self.tokenizer = tokenizer

    def language_reward(self, prompts, completions, **reward_kwargs) -> List[float]:
        language = 'en'
        return  [int(language==language_detection(text)) for text in completions]

    # weight = 0.1
    def has_think(self, prompts, completions, **reward_kwargs) -> List[float]:
        return  [int(text.count("</think>") >= 1) for text in completions]

    # weight = 0.1
    def has_judge(self, prompts, completions, **reward_kwargs) -> List[float]:
        return [int(len(re.findall(r'<judge>.*?</judge>', text))==1) for text in completions]

    def has_violation(self, prompts, completions, **reward_kwargs) -> List[float]:
        return [int(len(re.findall(r'<violations>.*?</violations>', text))==1) for text in completions]

    # weight = 0.1
    def has_query(self, prompts, completions, **reward_kwargs) -> List[float]:
        return [int(len(re.findall(r'<query>.*?</query>', text))>=1) for text in completions]

    # weight = 0.1
    def think_before_judge(self, prompts, completions, **reward_kwargs) -> List[float]:
        rewards = []
        # 遍历每个 completion，计算奖励
        for text in completions:
            reward = 0.0
            # 计算 <think> 和 <answer> 标签的数量
            if text.count("</think>") < 1:
                reward = 0.0
            elif len(re.findall(r'<judge>.*?</judge>', text)) != 0 and text.index("</think>") < text.index("<judge>"):
                reward = 1.0

            rewards.append(reward)
        return [r for r in rewards]

    # weight = 0.2
    def cot_length_reward(self, prompts, completions, **reward_kwargs) -> List[float]:
        rewards = []

        # 遍历每个 completion，计算奖励
        for p, c in zip(prompts, completions):
            if "</think>" in c:
                think = c.split("</think>")[0]
                think_length = len(self.tokenizer.tokenize(think))
                reward = math.sin(think_length / (self.config['best_cot_length'] * 2) * math.pi)
                rewards.append(reward)
            else:
                rewards.append(0.0)

        return [r for r in rewards]

    # weight = 0.2
    def repetition_think_panalty(self, prompts, completions, **reward_kwargs):

        def trigram_repetition_rate(sentence):
            # 分词（假设已经有分词工具）
            words = sentence.split()

            # 提取trigrams
            trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]

            # 计算trigrams频率
            trigram_counts = Counter(trigrams)

            # 计算重复的trigrams数量
            num_repeated = sum(count > 1 for count in trigram_counts.values())

            # 计算重复率
            repetition_rate = num_repeated / len(trigrams) if trigrams else 0
            return repetition_rate

        rewards = []

        for p, c in zip(prompts, completions):
            if "</think>" in c:
                think = c.split("</think>")[0]
                repetition_rate = trigram_repetition_rate(think)
                reward = math.sin((repetition_rate-2)/2*math.pi)+1 # math.sin((x-2)/2*math.pi)+1 三角函数
                rewards.append(reward)
            else:
                rewards.append(0.0)

        return [r for r in rewards]

    # weight = 0.4
    def acc_r(self, completions, ground_truth, **reward_kwargs) -> List[float]:
        # print(reward_kwargs.keys())
        rewards = []

        for text, g_t in zip(completions, ground_truth):
            judge = re.findall(r'<judge>(.*?)</judge>', text)
            if len(judge) != 0:
                judge = judge[0]
            else:
                judge = ''

            if judge.lower() == g_t.lower():
                reward = 1.0
            else:
                reward = 0.0
            rewards.append(reward)
        print("acc_r", rewards)
        return [r for r in rewards]
