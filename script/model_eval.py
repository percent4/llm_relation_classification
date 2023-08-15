# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: model_eval.py
# @time: 2023/8/15 19:08
# 模型评估脚本
import re
import pandas as pd
import requests
import json
from sklearn.metrics import classification_report, accuracy_score


# request for people relation classification
def get_predict_result(text, people1, people2):

    url = "http://10.241.132.208:5000/people_rel_cls"

    payload = json.dumps({
      "text": text,
      "people1": people1,
      "people2": people2
    })
    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.json()["result"]


if __name__ == '__main__':
    test_df = pd.read_csv('../data/rbert_test.csv', delimiter='\t')

    true_labels = []
    pred_labels = []
    # for i in range(50):
    for i in range(test_df.shape[0]):
        true_label, content = test_df.iloc[i, :].tolist()
        people1 = re.findall('<e1>(.+?)</e1>', content)[0]
        people2 = re.findall('<e2>(.+?)</e2>', content)[0]
        text = content.replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', '')
        pred_label = get_predict_result(text, people1, people2)
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        print(i, true_label, pred_label, text, people1, people2)

    # print(accuracy_score(true_labels, pred_labels))
    print(classification_report(true_labels, pred_labels, digits=4))
