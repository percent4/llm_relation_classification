# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: single_chat_server.py
# @time: 2023/7/25 22:27
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# 单轮对话web服务
from flask import Flask, request, jsonify

app = Flask("single_chat_server")


@app.route('/people_rel_cls', methods=['POST'])
def predict():
    req_dict = request.json
    text, people1, people2 = req_dict["text"], req_dict["people1"], req_dict["people2"]
    text = text.strip()
    content = f"给定以下标签：['不确定', '夫妻', '父母', '兄弟姐妹', '上下级', '师生', '好友', '同学', " \
              f"'合作', '同一个人', '情侣', '祖孙', '同门', '亲戚']，" \
              f"请在以下句子中分析并分类实体之间的关系：'{text}'" \
              f"在这个句子中，{people1}和{people2}之间的关系应该属于哪个标签？"
    content = "<s>{}</s>".format(content)
    print("content: ", content)
    input_ids = tokenizer(content, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs)
    print("response: ", response)
    response = response.strip().replace(text, "").replace('</s>', "").replace('<s>', "").strip()
    return jsonify({"result": response})


if __name__ == '__main__':
    model_name = "~/Firefly/script/checkpoint/firefly-baichuan-7b-people-merge"
    max_new_tokens = 5
    top_p = 0.9
    temperature = 0.01
    repetition_penalty = 1.0
    device = 'cuda:0'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    print("model loaded!")
    app.run(host="0.0.0.0", port=5000, threaded=True)
