import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

# 1. 基础模型路径和LoRA权重路径
base_model_path = "/home/ma-user/work/Qwen2.5-3B-Instruct"
lora_model_path = "./output-lora-train5000"

# 2. 选择设备
device = "npu" if hasattr(torch, 'npu') and torch.npu.is_available() else "cpu"

# 3. 加载Tokenizer和基座模型
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32
)

# 4. 加载LoRA权重
model = PeftModel.from_pretrained(model, lora_model_path)
model = model.to(device)
model.eval()

# 5. 多个问题列表
prompts = [
    "\ndef climbing_stairs(n: int) -> int:\n    \"\"\" You are climbing a staircase. It takes n steps to reach the top.\n\n    Each time you can either climb 1 or 2 steps. In how many distinct ways can you\n    climb to the top?\n\n    Example 1:\n        Input: n = 2\n        Output: 2\n        Explanation: There are two ways to climb to the top.\n            1. 1 step + 1 step\n            2. 2 steps\n    \n    Example 2:\n        Input: n = 3\n        Output: 3\n        Explanation: There are three ways to climb to the top.\n            1. 1 step + 1 step + 1 step\n            2. 1 step + 2 steps\n            3. 2 steps + 1 step\n    \n    Constraints:\n        1 <= n <= 45\n        \n    >>> climbing_stairs(2)\n    2\n    >>> climbing_stairs(3)\n    3\n    \"\"\"\n请以<think>推理过程</think><answer>最终答案</answer>的格式输出。",
    "\ndef sqrt_funtion_impl(x: int) -> int:\n    \"\"\" Given a non-negative integer x, return the square root of x rounded down to the \n    nearest integer. The returned integer should be non-negative as well.\n\n    You must not use any built-in exponent function or operator.\n    - For example, do not use pow(x, 0.5) in c++ or x ** 0.5 in python.\n    \n    Example 1:\n        Input: x = 4\n        Output: 2\n        Explanation: The square root of 4 is 2, so we return 2.\n    \n    Example 2:\n        Input: x = 8\n        Output: 2\n        Explanation: The square root of 8 is 2.82842..., and since we round it down to the\n        nearest integer, 2 is returned.\n        \n    Constraints:\n        0 <= x <= 2^31 - 1\n    \n    >>> sqrt_funtion_impl(4)\n    2\n    >>> sqrt_funtion_impl(8)\n    2\n    \"\"\"\n请以<think>推理过程</think><answer>最终答案</answer>的格式输出。",
    "\ndef add_binary(a: str, b: str) -> str:\n    \"\"\" Given two binary strings a and b, return their sum as a binary string.\n\n    Example 1:\n        Input: a = \"11\", b = \"1\"\n        Output: \"100\"\n    \n    Example 2:\n        Input: a = \"1010\", b = \"1011\"\n        Output: \"10101\"\n\n    Constraints:\n        1 <= a.length, b.length <= 104\n        a and b consist only of '0' or '1' characters.\n        Each string does not contain leading zeros except for the zero itself.\n        \n    >>> add_binary(\"11\", \"1\")\n    \"100\"\n    >>> add_binary(\"1010\", \"1011\")\n    \"10101\"\n    \"\"\"\n请以<think>推理过程</think><answer>最终答案</answer>的格式输出。",
    "from typing import List\n\n\ndef large_integer_plus_one(digits: List[int]) -> List[int]:\n    \"\"\" Increment the large integer by one and return the resulting array of digits.\n    \n    You are given a large integer represented as an integer array digits, where \n    each digits[i] is the ith digit of the integer. The digits are ordered from most \n    significant to least significant in left-to-right order. The large integer does \n    not contain any leading 0's.\n\n    Example 1:\n        Input: digits = [1,2,3]\n        Output: [1,2,4]\n        Explanation: The array represents the integer 123.\n        Incrementing by one gives 123 + 1 = 124.\n        Thus, the result should be [1,2,4].\n    \n    Example 2:\n        Input: digits = [4,3,2,1]\n        Output: [4,3,2,2]\n        Explanation: The array represents the integer 4321.\n        Incrementing by one gives 4321 + 1 = 4322.\n        Thus, the result should be [4,3,2,2].\n    \n    Example 3:\n        Input: digits = [9]\n        Output: [1,0]\n        Explanation: The array represents the integer 9.\n        Incrementing by one gives 9 + 1 = 10.\n        Thus, the result should be [1,0].\n    \n    Constraints:\n        1 <= digits.length <= 100\n        0 <= digits[i] <= 9\n        digits does not contain any leading 0's.\n        \n    >>> large_integer_plus_one([1,2,3])\n    [1,2,4]\n    >>> large_integer_plus_one([4,3,2,1])\n    [4,3,2,2]\n    >>> large_integer_plus_one([9])\n    [1,0]\n    \"\"\"\n请以<think>推理过程</think><answer>最终答案</answer>的格式输出。",
    "\ndef length_of_last_word(s: str) -> int:\n    \"\"\" Given a string s consisting of words and spaces, return the length of the \n    last word in the string.\n\n    A word is a maximal substring consisting of non-space characters only.\n\n    Example 1:\n        Input: s = \"Hello World\"\n        Output: 5\n        Explanation: The last word is \"World\" with length 5.\n    \n    Example 2:\n        Input: s = \"   fly me   to   the moon  \"\n        Output: 4\n        Explanation: The last word is \"moon\" with length 4.\n    \n    Example 3:\n        Input: s = \"luffy is still joyboy\"\n        Output: 6\n        Explanation: The last word is \"joyboy\" with length 6.\n    \n    Constraints:\n        1 <= s.length <= 104\n        s consists of only English letters and spaces ' '.\n        There will be at least one word in s.\n    \n    >>> length_of_last_word(\"Hello World\")\n    5\n    >>> length_of_last_word(\"   fly me   to   the moon  \")\n    4\n    >>> length_of_last_word(\"luffy is still joyboy\")\n    6\n    \"\"\"\n请以<think>推理过程</think><answer>最终答案</answer>的格式输出。"
]

# 6. 循环逐题推理
for i, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,    # 可调
            do_sample=False
        )
    end_time = time.time()

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"题目{i+1}：")
    print(result)
    print(f"推理耗时: {end_time - start_time:.3f} 秒")
    print("-" * 40)
