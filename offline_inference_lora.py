import json
import re
import os
import time
import argparse

# 设置环境变量强制离线模式，避免连接 HuggingFace Hub
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from logger import PerformanceLogger
from config import get_model_path, DEFAULT_DATA_PATH, DEFAULT_GENERATION_CONFIG, list_available_models, needs_trust_remote_code, clean_log_files

class Main:
    def __init__(self, model_path, data_path, adapter_path=None):
        self.model_path = model_path
        self.data_path = data_path
        self.adapter_path = adapter_path
        
        # 强制使用NPU设备
        if hasattr(torch, 'npu') and torch.npu.is_available():
            self.device = "npu:0"
            print(f"🚀 强制使用NPU设备: {self.device}")
            # 设置NPU为默认设备
            torch.npu.set_device(0)
        else:
            raise RuntimeError("NPU设备不可用！请检查NPU环境配置。")
        
        # 创建模型和tokenizer，根据配置决定是否使用trust_remote_code
        trust_remote_code = needs_trust_remote_code(model_path)
        if trust_remote_code:
            print(f"模型 {model_path} 需要trust_remote_code=True")
        
        try:
            print(f"🔄 加载tokenizer: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            print(f"🔄 加载基础模型到NPU: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # 使用float16以节省NPU内存
                device_map={"": self.device}  # 直接加载到NPU设备
            )
            
            # 如果指定了adapter路径，加载LoRA adapter
            if adapter_path:
                print(f"🔄 加载LoRA adapter到NPU: {adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model, 
                    adapter_path,
                    torch_dtype=torch.float16,
                    device_map={"": self.device}
                )
                print("✅ LoRA adapter加载完成")
            
            # 验证所有参数都在NPU设备上
            print(f"🔄 验证模型参数设备状态...")
            misplaced_params = []
            for name, param in self.model.named_parameters():
                if param.device != torch.device(self.device):
                    misplaced_params.append((name, param.device))
                    param.data = param.data.to(self.device)
                    if param.grad is not None:
                        param.grad.data = param.grad.data.to(self.device)
            
            if misplaced_params:
                print(f"⚠️  发现 {len(misplaced_params)} 个参数不在正确设备上，已自动修正")
                for name, device in misplaced_params[:5]:  # 只显示前5个
                    print(f"    {name}: {device} -> {self.device}")
            else:
                print(f"✅ 所有模型参数都在正确设备上")
            
            # 确保tokenizer的特殊token设置正确
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"🔧 设置pad_token为eos_token: {self.tokenizer.pad_token}")
            
            # 最终设备确认
            model_device = next(self.model.parameters()).device
            print(f"✅ 模型加载完成，设备: {model_device}")
            
            # 确保设备一致性
            if str(model_device) != self.device:
                print(f"⚠️  设备不一致，期望: {self.device}, 实际: {model_device}")
                raise RuntimeError(f"模型设备 {model_device} 与期望设备 {self.device} 不一致！")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            if trust_remote_code:
                print(f"模型 {model_path} 需要 trust_remote_code=True，请确保模型文件完整且支持离线加载")
            raise
        
        # 初始化性能日志器
        model_display_name = f"{model_path}"
        if adapter_path:
            model_display_name += f"+{adapter_path}"
        self.logger = PerformanceLogger(model_path=model_display_name)
        
        # 生成参数设置
        self.generation_params = {
            "choice": {
                "max_new_tokens": DEFAULT_GENERATION_CONFIG["max_tokens"]["choice"], 
                "temperature": DEFAULT_GENERATION_CONFIG["temperature"], 
                "top_p": DEFAULT_GENERATION_CONFIG["top_p"],
                "do_sample": True,
            },
            "code-generate": {
                "max_new_tokens": DEFAULT_GENERATION_CONFIG["max_tokens"]["code-generate"], 
                "temperature": DEFAULT_GENERATION_CONFIG["temperature"], 
                "top_p": DEFAULT_GENERATION_CONFIG["top_p"],
                "do_sample": True,
                "num_return_sequences": 3,  # 生成3个候选答案
            },
            "generic-generate": {
                "max_new_tokens": DEFAULT_GENERATION_CONFIG["max_tokens"]["generic-generate"], 
                "temperature": DEFAULT_GENERATION_CONFIG["temperature"], 
                "top_p": DEFAULT_GENERATION_CONFIG["top_p"],
                "do_sample": True,
            },
            "math": {
                "max_new_tokens": DEFAULT_GENERATION_CONFIG["max_tokens"]["math"], 
                "temperature": DEFAULT_GENERATION_CONFIG["temperature"], 
                "top_p": DEFAULT_GENERATION_CONFIG["top_p"],
                "do_sample": True,
            }
        }
        
        # NPU内存管理
        self.enable_memory_cleanup = True
        
        self.prompt_templates = {
            "choice": (
                "Answer the following multiple choice question. The last line of your response should be of the following format: "
                "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"
                "{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}\n"
                "请以<think>推理过程</think><answer>最终答案</answer>的格式输出。"
            ),
            "code-generate": (
                "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n"
                "{Question}\n"
                "请以<think>推理过程</think><answer>最终代码</answer>的格式输出。"
            ),
            "generic-generate": (
                "You will be asked to read a passage and answer a question. Think step by step, then write a line of the form 'Answer: $ANSWER' at the end of your response.\n"
                "{Question}\n"
                "请以<think>推理过程</think><answer>最终答案</answer>的格式输出。"
            ),
            "math": (
                "Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.\n\n"
                "{Question}\n\n"
                "Remember to put your answer on its own line after 'Answer:', and indicate your final answer in boxed LaTeX. For example, if the final answer is \\sqrt{{3}}, write it as \\boxed{{\\sqrt{{3}}}}。\n"
                "请以<think>推理过程</think><answer>最终答案</answer>的格式输出。"
            ),
        }

    def build_prompt(self, q):
        qtype = q['type']
        if qtype == 'choice':
            choices = q.get('choices', {})
            return self.prompt_templates['choice'].format(
                Question=q['prompt'],
                A=choices.get('A', ''),
                B=choices.get('B', ''),
                C=choices.get('C', ''),
                D=choices.get('D', ''),
            )
        elif qtype in self.prompt_templates:
            return self.prompt_templates[qtype].format(Question=q['prompt'])
        else:
            # 兜底
            return q['prompt']

    def clean_content(self, text):
        # 移除think标签前的多余字符
        match = re.search(r'<think>', text)
        if match:
            text = text[match.start():]
        return text.strip()
    
    def cleanup_memory(self):
        # 清理NPU内存
        if self.enable_memory_cleanup and hasattr(torch, 'npu'):
            torch.npu.empty_cache()
            torch.npu.synchronize()
            print("    🧹 NPU内存已清理")
    
    # 处理生成的文本
    def process_generated_texts(self, generated_texts, question_type):
        if question_type == "code-generate":
            # 对于代码生成，返回所有3个候选答案
            return [self.clean_content(text) for text in generated_texts]
        else:
            # 对于其他类型，返回第一个答案
            return self.clean_content(generated_texts[0]) if generated_texts else ""

    # 按类型批量处理题目
    def process_batch_by_type(self, questions_by_type):
        all_results = []
        
        for qtype, questions in questions_by_type.items():
            if not questions:
                continue
            
            question_count = len(questions)
            
            # 记录批次开始
            batch_start_time = self.logger.log_batch_start(qtype, question_count)
            
            generation_start = time.time()
            
            # 逐个处理题目（transformers更适合单个处理）
            for i, q_data in enumerate(questions):
                print(f"  📝 正在处理第 {i+1}/{len(questions)} 个 {qtype} 类型题目 (ID: {q_data['question']['id']})...")
                
                # 构建prompt
                prompt = self.build_prompt(q_data['question'])
                
                try:
                    # Tokenize输入
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                    input_length = inputs.input_ids.shape[1]
                    print(f"    🔤 输入长度: {input_length} tokens")
                    
                    # 将输入数据移动到NPU设备
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    print(f"    🔍 输入已移动到NPU设备: {self.device}")
                    
                    print(f"    🚀 开始生成...")
                    generation_start_single = time.time()
                    
                    # 确保生成参数中的设备相关设置
                    gen_params = self.generation_params[qtype].copy()
                    
                    # 动态设置pad_token_id，确保与输入在同一设备
                    if self.tokenizer.pad_token_id is not None:
                        gen_params["pad_token_id"] = self.tokenizer.pad_token_id
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            **gen_params
                        )
                    
                    generation_time_single = time.time() - generation_start_single
                    print(f"    ✅ 生成完成，耗时: {generation_time_single:.2f}s")
                    
                    # 解码生成的文本
                    if qtype == "code-generate":
                        # 对于代码生成，返回多个候选答案
                        generated_texts = []
                        for output in outputs:
                            # 只解码新生成的tokens
                            generated_tokens = output[input_length:]
                            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                            generated_texts.append(generated_text.strip())
                    else:
                        # 对于其他类型，返回单个答案
                        generated_tokens = outputs[0][input_length:]
                        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        generated_texts = [generated_text.strip()]
                    
                except Exception as e:
                    print(f"⚠️  生成第 {i+1} 个prompt时出错: {e}")
                    # 使用默认错误消息作为后备
                    if qtype == "code-generate":
                        generated_texts = ["# 生成失败", "# 生成失败", "# 生成失败"]
                    else:
                        generated_texts = ["生成失败，请检查输入"]
                
                # 处理结果
                processed_content = self.process_generated_texts(generated_texts, qtype)
                
                all_results.append({
                    "id": q_data['question']['id'], 
                    "content": processed_content
                })
                
                # 记录单个问题完成
                has_multiple = isinstance(processed_content, list)
                self.logger.log_question_complete(
                    q_data['question']['id'], qtype, has_multiple
                )
                
                # 清理NPU内存
                self.cleanup_memory()
            
            generation_time = time.time() - generation_start
            
            # 记录批次结束
            self.logger.log_batch_end(qtype, question_count, batch_start_time, generation_time)
        
        return all_results


    def infer(self):
        try:
            # 清理NPU内存
            print("🧹 开始前清理NPU内存...")
            self.cleanup_memory()
            
            # 开始计时
            self.logger.start_timing()
            
            # 每次都重新开始，不进行断点续传
            results = []
            
            # 加载全部题目
            with open(self.data_path, "r", encoding="utf-8") as fin:
                all_lines = [json.loads(line) for line in fin]
            
            # 按类型分组所有题目
            questions_by_type = {
                "choice": [],
                "code-generate": [],
                "generic-generate": [],
                "math": []
            }
            
            for idx, q in enumerate(all_lines, 1):
                qtype = q.get('type', 'generic-generate')
                if qtype in questions_by_type:
                    questions_by_type[qtype].append({
                        'question': q,
                        'idx': idx
                    })
            
            # 统计待处理题目
            total_pending = sum(len(questions) for questions in questions_by_type.values())
            
            self.logger.logger.info(f"开始处理 {total_pending} 道题目...")
            self.logger.logger.info(f"输出文件: {self.logger.output_path}")
            
            # 按类型批量处理
            new_results = self.process_batch_by_type(questions_by_type)
            results.extend(new_results)
            
            # 保存结果
            final_result = {"result": {"results": results}}
            self.logger.save_results(results)
            
            # 记录性能统计
            total_questions = len(results)
            self.logger.log_performance_summary(total_questions)
            
            self.logger.logger.info(f"全部处理完毕，共处理 {total_questions} 个题目")
            
            return final_result
            
        except Exception as e:
            self.logger.log_error(f"处理过程中发生错误: {e}")
            raise
        finally:
            # 确保日志器正确关闭
            self.logger.close()

    # 析构函数，确保资源释放
    def __del__(self):
        if hasattr(self, 'logger'):
            self.logger.close()

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="华为考试数据处理脚本")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="模型键名或完整路径 (如: qwen2.5-7b, qwen2-7b, 或完整路径)"
    )
    
    parser.add_argument(
        "--data", 
        type=str, 
        default=DEFAULT_DATA_PATH,
        help=f"数据文件路径 (默认: {DEFAULT_DATA_PATH})"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_GENERATION_CONFIG["temperature"],
        help=f"生成温度 (默认: {DEFAULT_GENERATION_CONFIG['temperature']})"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=DEFAULT_GENERATION_CONFIG["top_p"],
        help=f"Top-p 采样参数 (默认: {DEFAULT_GENERATION_CONFIG['top_p']})"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出所有可用的模型键名"
    )
    
    parser.add_argument(
        "--clean-logs",
        action="store_true",
        help="清理所有日志文件和统计文件"
    )
    
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="LoRA adapter路径 (如: ./output-lora-qwen)"
    )
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 如果用户要求列出模型，则显示后退出
    if args.list_models:
        list_available_models()
        return
    
    # 如果用户要求清理日志文件，则执行清理后退出
    if args.clean_logs:
        clean_log_files()
        return
    
    # 获取实际的模型路径
    model_path = get_model_path(args.model)
    
    print(f"🤖 使用模型: {model_path}")
    if args.adapter_path:
        print(f"🧩 LoRA Adapter: {args.adapter_path}")
    print(f"📁 数据文件: {args.data}")
    print(f"🌡️  温度参数: {args.temperature}")
    print(f"🎯 Top-p参数: {args.top_p}")
    print("="*50)
    
    # 创建主程序实例
    main_instance = Main(
        model_path=model_path,
        data_path=args.data,
        adapter_path=args.adapter_path
    )
    
    # 如果需要，可以动态更新生成参数
    if args.temperature != DEFAULT_GENERATION_CONFIG["temperature"] or args.top_p != DEFAULT_GENERATION_CONFIG["top_p"]:
        print(f"📝 更新生成参数: temperature={args.temperature}, top_p={args.top_p}")
        for qtype in main_instance.generation_params:
            main_instance.generation_params[qtype]["temperature"] = args.temperature
            main_instance.generation_params[qtype]["top_p"] = args.top_p
    
    # 开始推理
    main_instance.infer()

if __name__ == "__main__":
    main()
