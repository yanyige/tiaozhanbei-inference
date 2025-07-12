from datasets import load_dataset
import json
import re

# 加载原始训练数据集
print("🔄 加载bespokelabs/Bespoke-Stratos-17k数据集...")
dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k")



def format_output_with_think_answer(original_response, question_type):
    """
    将原始回答格式化为<think>思考过程</think><answer>最终结果</answer>格式
    根据题目类型调整answer部分的格式
    """
    # 如果原始回答包含 <|begin_of_thought|> 和 <|end_of_thought|> 格式
    if "<|begin_of_thought|>" in original_response and "<|end_of_thought|>" in original_response:
        # 提取思考过程
        start_idx = original_response.find("<|begin_of_thought|>") + len("<|begin_of_thought|>")
        end_idx = original_response.find("<|end_of_thought|>")
        thinking_process = original_response[start_idx:end_idx].strip()
        
        # 提取最终答案（在solution部分）
        if "<|begin_of_solution|>" in original_response and "<|end_of_solution|>" in original_response:
            start_sol = original_response.find("<|begin_of_solution|>") + len("<|begin_of_solution|>")
            end_sol = original_response.find("<|end_of_solution|>")
            solution_content = original_response[start_sol:end_sol].strip()
        else:
            # 如果没有solution部分，使用thought后面的内容
            solution_content = original_response[original_response.find("<|end_of_thought|>") + len("<|end_of_thought|>"):].strip()
    
    # 如果原始回答已经有明确的思考过程，尝试提取
    elif "解析" in original_response or "分析" in original_response or "步骤" in original_response:
        # 尝试分离思考过程和最终答案
        parts = original_response.split("答案")
        if len(parts) > 1:
            thinking_process = parts[0].strip()
            solution_content = parts[1].strip()
        else:
            # 如果没有明确的"答案"分隔，将前80%作为思考过程
            split_point = int(len(original_response) * 0.8)
            thinking_process = original_response[:split_point].strip()
            solution_content = original_response[split_point:].strip()
    else:
        # 根据题目类型生成思考过程
        if question_type == "code-generate":
            thinking_process = "我需要分析这个编程问题，理解需求并实现相应的算法。"
        elif question_type == "math":
            thinking_process = "我需要分析这个数学问题，找出解题思路并逐步计算。"
        elif question_type == "choice":
            thinking_process = "我需要分析各个选项，找出正确答案。"
        else:
            thinking_process = "我需要理解问题并提供准确的回答。"
        
        solution_content = original_response.strip()
    
    # 根据题目类型格式化最终答案
    formatted_answer = format_answer_by_type(solution_content, question_type)
    
    # 格式化输出
    formatted_output = f"<think>{thinking_process}</think><answer>{formatted_answer}</answer>"
    return formatted_output

def format_answer_by_type(solution_content, question_type):
    """
    根据题目类型格式化答案内容，确保符合评判正则表达式要求
    - choice: r"(?i)Answer[ \t]*:[ \t]*\$?([A-D]*)\$"
    - code-generate: r"```python\n(.*?)```"
    - math: r"(?i)Answer\s*:\s*\s*([^\n]+)"
    - generic-generate: r"(?i)Answer\s*:\s*([^\n]+)"
    """
    import re
    
    if question_type == "choice":
        # choice类型：检查是否已有"Answer: A"格式
        if re.search(r'(?i)Answer[ \t]*:[ \t]*\$?([A-D]*)\$?', solution_content):
            # 已有符合要求的格式，直接返回
            return solution_content.strip()
        
        # 没有符合要求的格式，尝试提取答案并添加
        patterns = [
            r'\\boxed\{([ABCDE])\}',  # \boxed{A}
            r'\\boxed\{\(([ABCDE])\)\}',  # \boxed{(A)}
            r'答案[是为：:]\s*([ABCDE])',  # 答案是A
            r'选择\s*([ABCDE])',  # 选择A
            r'therefore.*answer.*is.*\(([ABCDE])\)',  # therefore the answer is (A)
            r'answer.*is.*\(([ABCDE])\)',  # answer is (A)
            r'answer.*is.*([ABCDE])',  # answer is A
            r'option.*([ABCDE])',  # option A
            r'\(([ABCDE])\)',  # (A)
        ]
        
        # 按优先级尝试匹配
        for pattern in patterns:
            match = re.search(pattern, solution_content, re.IGNORECASE | re.MULTILINE)
            if match:
                letter = match.group(1).upper()
                if letter in 'ABCDE':
                    return f"{solution_content.strip()}\n\nAnswer: {letter}"
        
        # 如果都没找到，默认返回A（需要人工检查）
        return f"{solution_content.strip()}\n\nAnswer: A"
    
    elif question_type == "code-generate":
        # code-generate类型：检查是否已有```python代码块
        if re.search(r'```python\n(.*?)```', solution_content, re.DOTALL):
            # 已有符合要求的格式，直接返回
            return solution_content.strip()
        
        # 没有符合要求的格式，尝试提取代码并添加
        # 尝试提取通用代码块
        code_match = re.search(r'```\n?(.*?)```', solution_content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            return f"```python\n{code}\n```"
        
        # 尝试提取def开头的代码
        def_match = re.search(r'(def\s+\w+.*?)(?=\n\n|\n$|$)', solution_content, re.DOTALL)
        if def_match:
            code = def_match.group(1).strip()
            return f"```python\n{code}\n```"
        
        # 如果都没找到，把整个内容当作代码
        return f"```python\n{solution_content.strip()}\n```"
    
    elif question_type == "math":
        # math类型：检查是否已有"Answer: xxx"格式
        if re.search(r'(?i)Answer\s*:\s*([^\n]+)', solution_content):
            # 已有符合要求的格式，直接返回
            return solution_content.strip()
        
        # 没有符合要求的格式，尝试提取答案并添加
        # 尝试提取boxed内容作为答案
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution_content)
        if boxed_match:
            answer = boxed_match.group(1).strip()
            return f"{solution_content.strip()}\n\nAnswer: {answer}"
        
        # 尝试其他格式的答案
        answer_patterns = [
            r'答案[是为：:]\s*([^。\n]+)',  # 答案是xxx
            r'therefore.*answer.*is[：:]?\s*([^。\n]+)',  # therefore the answer is xxx
            r'final.*answer[：:]?\s*([^。\n]+)',  # final answer: xxx
            r'result[：:]?\s*([^。\n]+)',  # result: xxx
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, solution_content, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                return f"{solution_content.strip()}\n\nAnswer: {answer}"
        
        # 如果都没找到，尝试从最后一行提取数字或表达式
        lines = solution_content.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            # 查找数字、分数、表达式等
            if re.search(r'[\d\.\+\-\*\/\(\)\^\=\\]', line) and len(line) < 100:
                return f"{solution_content.strip()}\n\nAnswer: {line}"
        
        # 最后兜底
        return f"{solution_content.strip()}\n\nAnswer: 需要补充答案"
    
    elif question_type == "generic-generate":
        # generic-generate类型：直接在原始solution内容前加 Answer:
        # 检查是否已有"Answer: xxx"格式
        if re.search(r'(?i)Answer\s*:\s*([^\n]+)', solution_content):
            # 已有符合要求的格式，直接返回
            return solution_content.strip()
        
        # 没有符合要求的格式，直接在solution内容前加 Answer:
        return f"Answer: {solution_content.strip()}"
    
    else:
        # 默认情况，保持原内容
        return solution_content

def process_dataset(dataset):
    """
    处理数据集，转换为目标格式
    """
    processed_data = []
    
    # 获取训练数据
    train_data = dataset['train']
    total_samples = len(train_data)
    
    # 处理全量数据
    print(f"🔄 开始处理全部 {total_samples:,} 条训练数据...")
    
    for idx, row in enumerate(train_data):
        # 获取原始数据
        conversations = row.get('conversations', [])
        
        # 提取用户问题和助手回答
        original_prompt = ""
        original_response = ""
        
        for conv in conversations:
            if conv.get('from') == 'user':
                original_prompt = conv.get('value', '')
            elif conv.get('from') == 'assistant':
                original_response = conv.get('value', '')
        
        # 推测题目类型（如果没有明确类型字段）
        question_type = "generic-generate"  # 默认类型
        
        # 根据内容推测类型 - 修改后的逻辑
        
        # 1. 首先检查是否是选择题（优先级最高）
        has_choices = bool(re.search(r'\\text\{\\([A-E]\\)\}', original_prompt) or 
                          re.search(r'\\([A-E]\\)', original_prompt) or
                          re.search(r'\([A-E]\)', original_prompt) or
                          '(A)' in original_prompt or '(B)' in original_prompt or 
                          '(C)' in original_prompt or '(D)' in original_prompt or '(E)' in original_prompt or
                          'A)' in original_prompt or 'B)' in original_prompt or
                          'C)' in original_prompt or 'D)' in original_prompt or 'E)' in original_prompt)
        
        if has_choices:
            question_type = "choice"
        # 2. 然后检查是否是编程题（更严格的判断）
        elif ('def ' in original_prompt and 
              ('class' in original_prompt or 'import' in original_prompt or 'return' in original_prompt or
               'for ' in original_prompt or 'while ' in original_prompt or 'if ' in original_prompt)):
            question_type = "code-generate"
        # 3. 检查是否是数学题
        elif any(math_keyword in original_prompt.lower() for math_keyword in ['solve', 'calculate', 'equation', '计算', '求解', 'find', 'determine']):
            question_type = "math"
        # 4. 默认类型
        else:
            question_type = "generic-generate"
        
        # 格式化输出
        formatted_response = format_output_with_think_answer(original_response, question_type)
        
        # 创建新的数据条目
        new_entry = {
            "id": f"train_{idx:06d}",
            "type": question_type,
            "prompt": original_prompt,
            "response": formatted_response
        }
        
        processed_data.append(new_entry)
        
        # 进度提示
        if (idx + 1) % 1000 == 0 or idx + 1 == total_samples:
            progress = (idx + 1) / total_samples * 100
            print(f"  已处理 {idx + 1} / {total_samples} 条数据 ({progress:.1f}%)...")
    
    return processed_data

# 处理数据集
processed_training_data = process_dataset(dataset)

# 保存处理后的数据
output_file = "processed_training_data.jsonl"
print(f"🔄 保存处理后的数据到 {output_file}...")

with open(output_file, "w", encoding="utf-8") as f:
    for entry in processed_training_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ 成功处理并保存 {len(processed_training_data)} 条训练数据")
print(f"📁 输出文件: {output_file}")

# 显示一些样例
print("\n📋 处理后的数据样例:")
for i in range(min(3, len(processed_training_data))):
    sample = processed_training_data[i]
    print(f"\n样例 {i+1}:")
    print(f"类型: {sample['type']}")
    print(f"输入: {sample['prompt'][:100]}...")
    print(f"输出: {sample['response'][:200]}...")
