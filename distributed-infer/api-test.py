#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
import requests
import time
from tqdm import tqdm

# ================= 配置区域 =================
# 您的 API 地址
API_URL = "http://6.30.3.162:32607/v1/chat/completions"   #这里！！！！！！！！！
API_KEY = "deploy-wanghao"
# 模型名称
MODEL_NAME = "default" 

# 文件路径配置
TEST_FILE_PATH = "./test.json"
OUTPUT_EXCEL_PATH = './test-result-api.xlsx'

# Excel 列定义 (移到全局，方便循环内调用)
EXCEL_COLS = ['分类', 'session_id', 'turn', 'role', 'content', 'correct_flag', 'user_info']
# ===========================================

class JoyChatRequests:
    def __init__(self, api_url, model_name, api_key=""):
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key

    def generate_response(self, user_input, system_prompt, history):
        # 1. 构建 Messages
        messages = [{"role": "system", "content": system_prompt}]
        dialog = []
        for line in history:
            messages.append({"role": "user", "content": line[0]})
            messages.append({"role": "assistant", "content": line[1]})
            dialog.append({"role": "user", "content": line[0]})
            dialog.append({"role": "assistant", "content": line[1]})
        messages.append({"role": "user", "content": user_input})
        dialog.append({"role": "user", "content": user_input})

        # 2. 构造 Payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "n": 1,
            "presence_penalty": 0,
            "max_tokens": 4096,
            "stream": False
        }

        # 2.1 构造 Headers（新增）
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            # OpenAI 兼容服务最常见的方式
            headers["Authorization"] = f"Bearer {self.api_key}"
            # 如果你们服务用 X-API-Key，请改成下面这行之一：
            # headers["X-API-Key"] = self.api_key

        response_text = ""
        try:
            # 3. 发送请求（新增 headers 参数）
            resp = requests.post(self.api_url, json=payload, headers=headers)
            resp.raise_for_status()
            resp_json = resp.json()

            # 4. 解析 Response
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                message_obj = resp_json["choices"][0].get("message", {})
                response_text = message_obj.get("content", "")
            else:
                response_text = str(resp_json)
        except requests.HTTPError as e:
            tqdm.write(f"HTTP Error: {e} | status={resp.status_code} | body={resp.text[:500]}")
            response_text = ""
        except Exception as e:
            tqdm.write(f"Request Error: {e}")
            response_text = ""

        text_log = json.dumps(messages, ensure_ascii=False)
        return text_log, dialog, response_text

# ================= 主程序逻辑 =================

# 1. 读取数据
try:
    with open(TEST_FILE_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"错误：找不到测试文件 {TEST_FILE_PATH}")
    exit()

print(f"Total Data: {len(data)}")
print("=========================")

model = JoyChatRequests(API_URL, MODEL_NAME, API_KEY)

total = 0
correct = 0
diff = 0
out_result = []
out_excel = []

# 2. 进度条循环
pbar = tqdm(data, desc="Evaluating", unit="sample")

for messages in pbar:
    # 提取字段
    label = messages.get("output", "")
    system_prompt = messages.get("system", "")
    instructions = messages.get("instructions", "") 
    history = messages.get('history', [])
    session_id = messages.get('session_id', "")
    category = messages.get('category', "")

    if session_id == '新增':
        continue

    # 提取用户信息
    user_info = ""
    split_sys = system_prompt.split("\n")
    if len(split_sys) >= 3:
        if not '状态' in split_sys[-3]:
            user_info = split_sys[-3] + split_sys[-2]
        else:
            user_info = split_sys[-2]
    
    # === 调用 API ===
    input_text, dialog, response = model.generate_response(instructions, system_prompt, history)
    
    # === 结果评估 ===
    if response is None: response = ""
    
    if "</wmzy_state>" in response:
        state_pred = response.split("</wmzy_state>")[0].replace(" " ,"")
    else:
        state_pred = response.replace(" ", "")

    if "</wmzy_state>" in label:
        state_gth = label.split("</wmzy_state>")[0].replace(" ","")
    else:
        state_gth = label.replace(" ", "")

    correct_flag = 0

    if state_pred == state_gth:
        correct += 1
        correct_flag = 1
    else:
        diff += 1
        correct_flag = 0
        log_txt = (f"Mismatch! Pred: [{state_pred}] | Gth: [{state_gth}]")
        tqdm.write(log_txt)
        
        full_log = (input_text + "\n" + "pred label\t" + response + "\n" + 
                   "===========" + "\n" + "gth label\t" + label + "\n\n\n")
        out_result.append(full_log)

    # === 构建 Excel 数据 ===
    turn_index = 1
    for turn in dialog:
        out_excel.append([category, session_id, str(turn_index), turn['role'], turn['content'], "", ""])
        turn_index += 1
    
    out_excel.append([category, session_id, '', '预测结果', response, str(correct_flag), user_info])
    out_excel.append([category, session_id, '', '对比结果', label, "", ""])
    
    total += 1
    
    # === 新增：定期保存逻辑 ===
    if total > 0 and total % 100 == 0:
        try:
            pd_temp = pd.DataFrame(out_excel, columns=EXCEL_COLS)
            pd_temp.to_excel(OUTPUT_EXCEL_PATH, index=False)
            # 使用 tqdm.write 提示，不打断进度条
            tqdm.write(f">> Auto-saved at count {total} to {OUTPUT_EXCEL_PATH}")
        except Exception as e:
            tqdm.write(f"!! Warning: Auto-save failed (File might be open): {e}")

    # 更新进度条
    if total > 0:
        pbar.set_postfix({"Acc": f"{correct/total:.2%}", "Diff": diff})

# 3. 最终保存结果
try:
    pd_excel = pd.DataFrame(out_excel, columns=EXCEL_COLS)
    pd_excel.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print(f"\nFinal Result saved to {OUTPUT_EXCEL_PATH}")
except Exception as e:
    print(f"Final Save Excel Error: {e}")
    # 如果最终保存失败，尝试保存一个带时间戳的备份文件，防止数据丢失
    backup_path = f"{OUTPUT_EXCEL_PATH}.backup_{int(time.time())}.xlsx"
    pd_excel.to_excel(backup_path, index=False)
    print(f"Saved to backup: {backup_path}")

# 保存错误日志
if out_result:
    with open("./result.txt", "w", encoding='utf-8') as f:
        for item in out_result:
            f.write(item)

if total > 0:
    print(f"Final Accuracy: {correct/total:.4f}")