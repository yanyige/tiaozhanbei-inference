import json
import re
import os
import time
import argparse
import torch

# 设置环境变量强制离线模式，避免连接 HuggingFace Hub
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 导入华为昇腾NPU支持
try:
    import torch_npu
    print("华为昇腾NPU支持已加载")
except ImportError:
    print("警告: 未找到torch_npu，将使用CPU或GPU")

from transformers import AutoModelForCausalLM, AutoTokenizer
from logger import PerformanceLogger
from config import get_model_path, DEFAULT_DATA_PATH, DEFAULT_GENERATION_CONFIG, list_available_models, needs_trust_remote_code, clean_log_files

class Main:
    def __init__(self, model_path, data_path, device="auto", test_mode=False):
        self.model_path = model_path
        self.data_path = data_path
        self.test_mode = test_mode
        
        # 设备检测和配置
        self.device = self._get_device(device)
        print(f"🔧 使用设备: {self.device}")
        
        # 创建模型和tokenizer实例，根据配置决定是否使用trust_remote_code
        trust_remote_code = needs_trust_remote_code(model_path)
        if trust_remote_code:
            print(f"模型 {model_path} 需要trust_remote_code=True")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # 根据设备类型配置模型加载参数
            model_kwargs = {
                "trust_remote_code": True,
            }
            
            # 对于NPU，使用float32可能更稳定
            if self.device.startswith("npu"):
                model_kwargs["torch_dtype"] = torch.float32
                print("🔧 NPU设备使用float32精度")
            else:
                model_kwargs["torch_dtype"] = torch.float16  # 其他设备使用半精度
                print("🔧 使用float16半精度以节省内存")
            
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            
            # 手动移动模型到指定设备
            print(f"📦 正在将模型移动到设备: {self.device}")
            self.model = self.model.to(self.device)
            
            # 确保有pad_token_id
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            print("模型加载成功！")
        except Exception as e:
            print(f"加载模型失败: {e}")
            if trust_remote_code:
                print(f"模型 {model_path} 需要 trust_remote_code=True，请确保模型文件完整且支持离线加载")
            raise
        
        # 初始化性能日志器
        self.logger = PerformanceLogger(model_path=model_path)
        
        # 初始化提示模板
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
        
        # 采样参数设置（为稳定性优化）
        self.generation_params = {
            "choice": {
                "max_new_tokens": min(DEFAULT_GENERATION_CONFIG["max_tokens"]["choice"], 512), 
                "temperature": DEFAULT_GENERATION_CONFIG["temperature"], 
                "top_p": DEFAULT_GENERATION_CONFIG["top_p"],
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1  # 避免重复生成
            },
            "code-generate": {
                "max_new_tokens": min(DEFAULT_GENERATION_CONFIG["max_tokens"]["code-generate"], 1024), 
                "temperature": DEFAULT_GENERATION_CONFIG["temperature"], 
                "top_p": DEFAULT_GENERATION_CONFIG["top_p"],
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "num_return_sequences": 3,
                "repetition_penalty": 1.1
            },
            "generic-generate": {
                "max_new_tokens": min(DEFAULT_GENERATION_CONFIG["max_tokens"]["generic-generate"], 512), 
                "temperature": DEFAULT_GENERATION_CONFIG["temperature"], 
                "top_p": DEFAULT_GENERATION_CONFIG["top_p"],
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1
            },
            "math": {
                "max_new_tokens": min(DEFAULT_GENERATION_CONFIG["max_tokens"]["math"], 512), 
                "temperature": DEFAULT_GENERATION_CONFIG["temperature"], 
                "top_p": DEFAULT_GENERATION_CONFIG["top_p"],
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1
            }
        }
    
    # 检测和配置可用的设备
    def _get_device(self, device_preference="auto"):
        if device_preference == "auto":
            # 自动检测设备
            # 优先检测NPU
            if hasattr(torch, 'npu') and torch.npu.is_available():
                device_count = torch.npu.device_count()
                print(f"🚀 自动检测到 {device_count} 个NPU设备")
                return f"npu:0"  # 使用第一个NPU设备
            
            # 检测GPU
            elif torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                print(f"🚀 自动检测到 {device_count} 个GPU设备")
                return "cuda:0"  # 使用第一个GPU设备
            
            # 使用CPU
            else:
                print("⚠️  未检测到NPU或GPU，将使用CPU")
                return "cpu"
        else:
            # 用户指定设备
            if device_preference == "npu":
                if hasattr(torch, 'npu') and torch.npu.is_available():
                    print(f"🚀 使用用户指定的NPU设备")
                    return "npu:0"
                else:
                    print("❌ NPU不可用，回退到CPU")
                    return "cpu"
            elif device_preference == "cuda":
                if torch.cuda.is_available():
                    print(f"🚀 使用用户指定的GPU设备")
                    return "cuda:0"
                else:
                    print("❌ GPU不可用，回退到CPU")
                    return "cpu"
            else:
                print(f"🔧 使用用户指定的设备: {device_preference}")
                return device_preference

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
    
    # 处理生成的文本
    def process_generated_texts(self, generated_texts, question_type):
        if question_type == "code-generate":
            # 对于代码生成，返回所有3个候选答案
            return [self.clean_content(text) for text in generated_texts]
        else:
            # 对于其他类型，返回第一个答案
            return self.clean_content(generated_texts[0]) if generated_texts else ""

    # 按类型批量处理题目，每处理完一个题目就立即保存
    def process_batch_by_type(self, questions_by_type, all_results):
        
        for qtype, questions in questions_by_type.items():
            if not questions:
                continue
            
            question_count = len(questions)
            
            # 记录批次开始
            batch_start_time = self.logger.log_batch_start(qtype, question_count)
            
            # 逐个处理题目（支持实时保存）
            generation_start = time.time()
            
            for i, q_data in enumerate(questions):
                print(f"  📝 正在处理第 {i+1}/{len(questions)} 个 {qtype} 类型题目 (ID: {q_data['question']['id']})...")
                
                # 构建prompt
                prompt = self.build_prompt(q_data['question'])
                
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                    input_length = inputs.input_ids.shape[1]
                    print(f"    🔤 输入长度: {input_length} tokens")
                    
                    # 将输入数据移动到指定设备
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    print(f"    🚀 开始生成...")
                    generation_start_single = time.time()
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            **self.generation_params[qtype]
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
                
                # 创建结果对象
                new_result = {
                    "id": q_data['question']['id'], 
                    "content": processed_content
                }
                
                # 🔥 立即保存这个结果到JSON文件
                all_results = self.logger.save_single_result(new_result, all_results)
                
                # 记录单个问题完成
                has_multiple = isinstance(processed_content, list)
                self.logger.log_question_complete(
                    q_data['question']['id'], qtype, has_multiple
                )
                
                # 清理内存（针对NPU和GPU）
                if self.device.startswith("npu"):
                    try:
                        torch.npu.empty_cache()
                    except:
                        pass
                elif self.device.startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
            
            generation_time = time.time() - generation_start
            
            # 记录批次结束
            self.logger.log_batch_end(qtype, question_count, batch_start_time, generation_time)
        
        return all_results


    def infer(self):
        try:
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
                    # 测试模式：每种类型只取第一道题
                    if self.test_mode and len(questions_by_type[qtype]) >= 1:
                        continue
                    
                    questions_by_type[qtype].append({
                        'question': q,
                        'idx': idx
                    })
            
            # 统计待处理题目
            total_pending = sum(len(questions) for questions in questions_by_type.values())
            
            if self.test_mode:
                self.logger.logger.info("🧪 测试模式：每种类型只处理第一道题目")
            
            self.logger.logger.info(f"开始处理 {total_pending} 道题目...")
            self.logger.logger.info(f"输出文件: {self.logger.output_path}")
            print(f"💾 将实时保存结果到: {self.logger.output_path}")
            
            # 按类型批量处理（会实时保存每个结果）
            results = self.process_batch_by_type(questions_by_type, results)
            
            # 最终确认保存完整结果（所有结果已实时保存）
            final_result = {"result": {"results": results}}
            print(f"✅ 所有结果已实时保存完成，文件: {self.logger.output_path}")
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "npu"],
        help="指定使用的设备 (默认: auto - 自动检测)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出所有可用的模型键名"
    )
    
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="列出所有可用的设备"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="测试模式：只处理每种类型的第一道题目"
    )
    
    parser.add_argument(
        "--clean-logs",
        action="store_true",
        help="清理所有日志文件和统计文件"
    )
    
    return parser.parse_args()

# 列出所有可用的设备
def list_available_devices():
    print("可用的设备:")
    print("  cpu: CPU (总是可用)")
    
    # 检测NPU
    if hasattr(torch, 'npu') and torch.npu.is_available():
        device_count = torch.npu.device_count()
        print(f"  npu: 华为昇腾NPU ({device_count} 个设备)")
    else:
        print("  npu: 华为昇腾NPU (不可用)")
    
    # 检测GPU
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"  cuda: NVIDIA GPU ({device_count} 个设备)")
    else:
        print("  cuda: NVIDIA GPU (不可用)")
    
    print("  auto: 自动检测 (推荐)")
    print()
    
    # 显示当前推荐的设备
    if hasattr(torch, 'npu') and torch.npu.is_available():
        print("🚀 推荐使用: npu (华为昇腾NPU)")
    elif torch.cuda.is_available():
        print("🚀 推荐使用: cuda (NVIDIA GPU)")
    else:
        print("🚀 推荐使用: cpu")

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 如果用户要求列出模型，则显示后退出
    if args.list_models:
        list_available_models()
        return
    
    # 如果用户要求列出设备，则显示后退出
    if args.list_devices:
        list_available_devices()
        return
    
    # 如果用户要求清理日志文件，则执行清理后退出
    if args.clean_logs:
        clean_log_files()
        return
    
    # 获取实际的模型路径
    model_path = get_model_path(args.model)
    
    print(f"🤖 使用模型: {model_path}")
    print(f"📁 数据文件: {args.data}")
    print(f"🌡️  温度参数: {args.temperature}")
    print(f"🎯 Top-p参数: {args.top_p}")
    print(f"🔧 指定设备: {args.device}")
    if args.test_mode:
        print(f"🧪 测试模式: 启用")
    print("="*50)
    
    # 创建主程序实例
    main_instance = Main(
        model_path=model_path,
        data_path=args.data,
        device=args.device,
        test_mode=args.test_mode
    )
    
    # 如果需要，可以动态更新采样参数
    if args.temperature != DEFAULT_GENERATION_CONFIG["temperature"] or args.top_p != DEFAULT_GENERATION_CONFIG["top_p"]:
        print(f"📝 更新采样参数: temperature={args.temperature}, top_p={args.top_p}")
        for qtype in main_instance.generation_params:
            main_instance.generation_params[qtype]["temperature"] = args.temperature
            main_instance.generation_params[qtype]["top_p"] = args.top_p
    
    # 开始推理
    main_instance.infer()

if __name__ == "__main__":
    main()
