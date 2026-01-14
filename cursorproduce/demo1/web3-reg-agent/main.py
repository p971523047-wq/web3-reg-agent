from openai import OpenAI
import json

# 1. 读取 prompt
with open("prompt/extract_event.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# 2. 读取监管文本
with open("input/policy_raw.txt", "r", encoding="utf-8") as f:
    regulatory_text = f.read()

# 3. 调模型
client = OpenAI(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    model="gpt-4.1-mini",   # 或你实际用的模型
    temperature=0,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": regulatory_text}
    ]
)

raw_output = response.choices[0].message.content

# 4. 写结果
data = json.loads(raw_output)

with open("output/event.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
