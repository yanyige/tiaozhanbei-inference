import json
import re
import os
import time
import argparse

# 设置环境变量强制离线模式，避免连接 HuggingFace Hub
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from vllm import LLM, SamplingParams
from logger import PerformanceLogger
from config import get_model_path, DEFAULT_DATA_PATH, DEFAULT_GENERATION_CONFIG, list_available_models, needs_trust_remote_code, clean_log_files

class Main:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        
        # 创建LLM实例，根据配置决定是否使用trust_remote_code
        trust_remote_code = needs_trust_remote_code(model_path)
        if trust_remote_code:
            print(f"模型 {model_path} 需要trust_remote_code=True")
        
        try:
            self.llm = LLM(model=model_path, trust_remote_code=True)
        except Exception as e:
            print(f"加载模型失败: {e}")
            if trust_remote_code:
                print(f"模型 {model_path} 需要 trust_remote_code=True，请确保模型文件完整且支持离线加载")
            raise
        
        # 初始化性能日志器
        self.logger = PerformanceLogger(model_path=model_path)
        
        # 采样参数设置
        self.sampling_params = {
            "choice": SamplingParams(
                max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"]["choice"], 
                temperature=DEFAULT_GENERATION_CONFIG["temperature"], 
                top_p=DEFAULT_GENERATION_CONFIG["top_p"]
            ),
            "code-generate": SamplingParams(
                n=3,
                max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"]["code-generate"], 
                temperature=DEFAULT_GENERATION_CONFIG["temperature"], 
                top_p=DEFAULT_GENERATION_CONFIG["top_p"]
            ),
            "generic-generate": SamplingParams(
                max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"]["generic-generate"], 
                temperature=DEFAULT_GENERATION_CONFIG["temperature"], 
                top_p=DEFAULT_GENERATION_CONFIG["top_p"]
            ),
            "math": SamplingParams(
                max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"]["math"], 
                temperature=DEFAULT_GENERATION_CONFIG["temperature"], 
                top_p=DEFAULT_GENERATION_CONFIG["top_p"]
            )
        }
        
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
            
            # 准备该类型的所有prompts
            prompts = []
            for q_data in questions:
                prompt = self.build_prompt(q_data['question'])
                prompts.append(prompt)
            
            # 批量生成
            generation_start = time.time()
            outputs = self.llm.generate(prompts, self.sampling_params[qtype])
            generation_time = time.time() - generation_start
            
            # 处理结果
            for output, q_data in zip(outputs, questions):
                generated_texts = [o.text for o in output.outputs]
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
    print(f"📁 数据文件: {args.data}")
    print(f"🌡️  温度参数: {args.temperature}")
    print(f"🎯 Top-p参数: {args.top_p}")
    print("="*50)
    
    # 创建主程序实例
    main_instance = Main(
        model_path=model_path,
        data_path=args.data
    )
    
    # 如果需要，可以动态更新采样参数
    if args.temperature != DEFAULT_GENERATION_CONFIG["temperature"] or args.top_p != DEFAULT_GENERATION_CONFIG["top_p"]:
        print(f"📝 更新采样参数: temperature={args.temperature}, top_p={args.top_p}")
        for qtype in main_instance.sampling_params:
            main_instance.sampling_params[qtype].temperature = args.temperature
            main_instance.sampling_params[qtype].top_p = args.top_p
    
    # 开始推理
    main_instance.infer()

if __name__ == "__main__":
    main()
