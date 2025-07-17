from qwen import LLMEngine
from api import DeepSeekEngine
import os
import json
import re
import unicodedata
def sanitize_filename(title):
    filename = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
    filename = re.sub(r'[^\w\-_. ]', '_', filename)
    filename = filename.strip()
    filename = filename.replace(' ', '_')
    return filename[:255]

# export CUDA_VISIBLE_DEVICES=0 && python3 main.py
# watch -n 1 --color gpustat --color

current_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    engine = DeepSeekEngine(
        api_key="sk-5a2facdcfcfc48b2a1e9c87a21d1629f",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    # engine = LLMEngine(
    #     model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/Qwen2.5-7B-Instruct"),
    #     tensor_parallel_size=len(os.environ.get("CUDA_VISIBLE_DEVICES").split(','))
    # )
    max = 1
    for rt, dirs, fs in os.walk(os.path.join(current_dir, "data")):
        for dir in dirs:
            os.makedirs(os.path.join(current_dir, f"mask/{dir}/"), exist_ok=True)
            for root, drs, files in os.walk(os.path.join(rt, f"{dir}")):
                for file in files:
                    if file.lower().endswith('.json'):
                        max -= 1
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as file:
                                data = json.load(file)
                        except Exception as e:
                            print(f"读取失败 {file_path}: {str(e)}")
                            continue
                        new_data = {
                            "Case Information": data["Case Information"],
                            "Physical Examination": data["Physical Examination"],
                            "Diagnostic Tests": data["Diagnostic Tests"],
                            "Final Diagnosis": data["Final Diagnosis"],
                            "options": data["options"],
                            "right_option": data["right_option"]
                        }
                        new_data = json.dumps(new_data, separators=(',', ':'))
                        print(f"### 正在处理{data['periodical']} {data['id']}...")
                        images = data['images']
                        for image in images:
                            title = image['title']
                            caption = image['caption']
                            prompt = (f"任务目标：请根据提供的医学案例数据data:{new_data}，通过图片的title：{title}和caption：{caption}判断图片是否同时满足以下两个条件："
                                "1. 是案例中提及的有用图片（与data中的\"Case Information\"/\"Physical Examination\"/\"Diagnostic Tests\"/\"Final Diagnosis\"/\"options\"相关）"
                                "2. 不直接包含病理诊断答案（非病理性图片或免疫组化结果图）"
                                "判断标准："
                                "- 符合条件：临床检查图、影像学原图（MRI/CT等）、解剖示意图等"
                                "- 不符合条件：病理切片图、免疫组化染色图、直接显示诊断结论的示意图"
                                "3. 输出要求："
                                "请严格只输出合法 JSON 内容，禁止输出任何解释、说明、引导性语言或换行。直接返回形如："
                                "{\"related_to_text\": true, \"contains_diagnosis\": false, \"is_valid\": true, \"reason\": \"xxx\"}"
                                "不要包含多个 JSON。仅输出一个对象。"
                                "- 必须包含全部4个字段，布尔值全小写"
                                "- `reason` 字段用中文简要说明判断依据"
                                "示例："
                                "{\"related_to_text\": false,\"contains_diagnosis\": true,\"is_valid\": false,\"reason\": \"图片为HE染色切片，直接显示肿瘤细胞形态\"}")
                            response =engine.generate(prompt)
                            with open(os.path.join(current_dir, f"mask/{dir}/{sanitize_filename(title)}.txt"), 'w', encoding='utf-8') as file:
                                file.write(f"=== Prompt ===\n{prompt}\n\n=== Response ===\n{response}")
                            try:
                                match = re.search(r'\{.*?\}', response, re.DOTALL)
                                if match:
                                    json_str = match.group(0)
                                parsed = json.loads(json_str)
                                with open(os.path.join(current_dir, f"mask/{dir}/{sanitize_filename(title)}.json"), 'w', encoding='utf-8') as file:
                                    json.dump(parsed, file, indent=2, ensure_ascii=False)
                            except json.JSONDecodeError:
                                print("JSON解析失败")
                        break
            if max < 0:
                break
    return

if __name__ == "__main__":
    main()