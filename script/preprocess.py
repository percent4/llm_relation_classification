# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: preprocess.py
# @time: 2023/8/15 19:01
import re
import jsonlines
# make data into Firefly format

with open('../data/rbert_test.csv', 'r') as f:
    lines = f.readlines()


def make_message(text, people1, people2):
    # prompt created by GPT-4
    content = "给定以下标签：['不确定', '夫妻', '父母', '兄弟姐妹', '上下级', '师生', '好友', '同学', " \
              "'合作', '同一个人', '情侣', '祖孙', '同门', '亲戚']，" \
              f"请在以下句子中分析并分类实体之间的关系：'{text}'" \
              f"在这个句子中，{people1}和{people2}之间的关系应该属于哪个标签？"
    return content


for i, _ in enumerate(lines[1:]):
    label, content = _.strip().split('\t')
    people1 = re.findall('<e1>(.+?)</e1>', content)[0]
    people2 = re.findall('<e2>(.+?)</e2>', content)[0]
    text = content.replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', '')
    message = {"conversation_id": i+1,
               "category": "relation classification",
               "conversation": [
                    {
                        "human": make_message(text, people1, people2),
                        "assistant": label
                    }]
               }
    with jsonlines.open("../data/people.jsonl", 'a') as w:
        w.write(message)
