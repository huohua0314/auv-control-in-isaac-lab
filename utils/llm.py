import base64
import json
import requests
import os
import re
from datetime import datetime
from typing import List, Tuple

class VLAController:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", api_base: str = "https://halogg.cn/v1"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base.rstrip('/')
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # 确保日志目录存在
        self.log_dir = "vla_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _save_response_to_file(self, raw_content: str):
        """将大模型的原始输出追加到日志文件中，并附带时间戳"""
        # 1. 设定固定的日志文件名
        filename = os.path.join(self.log_dir, "all_responses.log")
        
        # 2. 生成当前记录的时间戳（精确到毫秒）
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        try:
            # 使用 "a" 模式进行追加写入
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Timestamp: {current_time}\n")
                f.write(f"Content:\n{raw_content}\n")
                f.write(f"{'='*50}\n")
                
            print(f"[VLA] Response appended to {filename}")
        except Exception as e:
            print(f"[VLA Warning] Failed to save log: {e}")

    def get_action(self, image_paths: List[str], task_description: str) -> str:
        """
        获取动作指令。
        返回：提取出的指令字符串（例如 "move forward"）
        """
        print(f"[VLA] Processing {len(image_paths)} camera views...")
        
        view_labels = ["Front", "Back", "Left", "Right", "Up", "Down"]
        content_list = []
        
        # 组装 Prompt
        from utils.prompt.dock import formatted_prompt
        prompt = formatted_prompt + "\n" + task_description
        content_list.append({"type": "text", "text": prompt})

        for path, label in zip(image_paths, view_labels):
            try:
                b64 = self._encode_image(path)
                content_list.append({"type": "text", "text": f"--- {label} View ---"})
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            except Exception as e:
                print(f"[VLA Warning] Failed to load {label} image: {e}")

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content_list}],
            "temperature": 0.2
            # 注意：移除了 json_object 格式限制，因为输出包含 $$ 标记
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions", 
                headers=self.headers, 
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            raw_content = result['choices'][0]['message']['content'].strip()
            
            # 1. 保存原始输出到文件
            self._save_response_to_file(raw_content)
            
            # 2. 提取最后的指令
            # 假设格式为: $$ Analysis @@@ Info $$ Instruction
            # 我们取最后一个 $$ 之后的内容，或者使用切片
            if "$$" in raw_content:
                # 寻找最后一个 $$ 的位置
                last_marker_index = raw_content.rfind("$$")
                # 提取 $$ 之后的部分并去除首尾空格及中文逗号
                instruction = raw_content[last_marker_index + 2:].strip()
                # 去除可能存在的中文逗号干扰
                instruction = instruction.replace("，", "").strip()
            else:
                # 如果没有找到标记，返回全文（作为保底）
                instruction = raw_content

            print(f"[VLA Success] Extracted Instruction: {instruction}")
            return instruction

        except Exception as e:
            print(f"[VLA Error] Inference failed: {e}")
            return "stop" # 发生错误时默认停止


