import logging
import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from config import LOG_CONFIG, DEFAULT_MODEL_PATH, LOGS_BASE_DIR, STATS_FILE_PATH


class PerformanceLogger:
    def __init__(self, model_path=None):
        # 如果没有指定模型路径，使用配置文件中的默认值
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
            
        self.model_path = model_path
        # 提取模型名称
        self.model_name = self._extract_model_name(model_path)
        
        # 生成序号和时间戳
        self.run_id = self._get_next_run_id()
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # 创建目录结构
        self._setup_directories()
        
        # 设置文件路径
        self._setup_file_paths()
        
        # 配置日志系统
        self._setup_logging()
        
        # 性能统计数据
        self.stats = {
            'batch_times': {},
            'start_time': None
        }
        
        # 记录开始信息
        self.logger.info(f"初始化性能记录器 - 模型: {self.model_name}, 运行序号: {self.run_id}")

    # 从模型路径提取模型名称
    def _extract_model_name(self, model_path):
        # 例如: "/home/ma-user/work/Qwen2.5-7B-Instruct" -> "Qwen2.5-7B-Instruct"
        model_name = Path(model_path).name
        # 清理文件名中的特殊字符
        model_name = model_name.replace("/", "_").replace("\\", "_")
        return model_name

    # 获取下一个运行序号（全局自增，跨模型）
    def _get_next_run_id(self):
        # 从performance_stats.csv文件中读取当前最大ID
        current_max_id = 0
        
        # 确保日志目录存在
        os.makedirs(LOGS_BASE_DIR, exist_ok=True)
        
        # 需要检查的文件列表（包括备份文件）
        files_to_check = [STATS_FILE_PATH]
        backup_file = STATS_FILE_PATH + '.backup'
        if os.path.exists(backup_file):
            files_to_check.append(backup_file)
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                run_id = int(row.get('run_id', '0'))
                                current_max_id = max(current_max_id, run_id)
                            except (ValueError, TypeError):
                                # 跳过无效的run_id
                                continue
                except (IOError, csv.Error):
                    # 如果文件读取失败，继续处理下一个文件
                    continue
        
        # 生成新的ID
        new_id = current_max_id + 1
        
        return f"{new_id:03d}"

    # 创建目录结构
    def _setup_directories(self):
        # 使用配置的日志目录
        self.logs_base_dir = LOGS_BASE_DIR
        
        # 模型特定目录
        self.model_logs_dir = os.path.join(self.logs_base_dir, self.model_name)
        
        # 创建目录
        os.makedirs(self.model_logs_dir, exist_ok=True)

    # 设置文件路径
    def _setup_file_paths(self):
        # 结果文件: logs/Qwen2.5-7B-Instruct/001-Qwen2.5-7B-Instruct-20241219-143022.json
        self.output_filename = f"{self.run_id}-{self.model_name}-{self.timestamp}.json"
        self.output_path = os.path.join(self.model_logs_dir, self.output_filename)
        
        # 统计文件: 使用配置的统计文件路径
        self.stats_file = STATS_FILE_PATH
        
        # 日志文件: logs/Qwen2.5-7B-Instruct/001-Qwen2.5-7B-Instruct-20241219-143022.log
        self.log_filename = f"{self.run_id}-{self.model_name}-{self.timestamp}.log"
        self.log_file = os.path.join(self.model_logs_dir, self.log_filename)

    # 配置日志系统
    def _setup_logging(self):
        # 创建logger
        self.logger = logging.getLogger(f"performance_logger_{self.run_id}")
        
        # 设置日志级别
        log_level = getattr(logging, LOG_CONFIG["level"].upper())
        self.logger.setLevel(log_level)
        
        # 清除现有的handlers
        self.logger.handlers.clear()
        
        # 文件handler - 详细日志
        file_handler = logging.FileHandler(self.log_file, encoding=LOG_CONFIG["encoding"])
        file_handler.setLevel(log_level)
        
        # 控制台handler - 简化输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 格式器
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter('%(message)s')
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        # 添加handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    # 开始计时
    def start_timing(self):
        self.stats['start_time'] = time.time()
        self.logger.info("开始处理...")

    # 记录批次开始
    def log_batch_start(self, question_type, count):
        self.logger.info(f"开始处理 {question_type} 类型题目，共 {count} 道...")
        return time.time()

    # 记录批次结束
    def log_batch_end(self, question_type, count, batch_start_time, generation_time):
        batch_total_time = time.time() - batch_start_time
        avg_time = batch_total_time / count if count > 0 else 0
        
        # 保存统计数据
        self.stats['batch_times'][question_type] = {
            'count': count,
            'total_time': batch_total_time,
            'avg_time': avg_time,
            'generation_time': generation_time
        }
        
        self.logger.info(
            f"完成 {question_type} 类型，{count} 道题，"
            f"总耗时 {batch_total_time:.2f}s，平均 {avg_time:.2f}s/题"
        )

    # 记录单个问题完成
    def log_question_complete(self, question_id, question_type, has_multiple_answers=False):
        if has_multiple_answers:
            self.logger.debug(f"完成题目 {question_id} ({question_type}, 多候选答案)")
        else:
            self.logger.debug(f"完成题目 {question_id} ({question_type})")

    # 保存结果文件
    def save_results(self, results):
        final_result = {"result": {"results": results}}
        
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存到: {self.output_path}")
    
    # 增量保存单个结果到JSON文件
    def save_single_result(self, new_result, all_results):
        # 将新结果添加到总结果列表
        all_results.append(new_result)
        
        # 立即保存到文件
        final_result = {"result": {"results": all_results}}
        
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"    💾 已保存题目 {new_result['id']} 的结果 ({len(all_results)} 题总计)")
        self.logger.debug(f"💾 已保存题目 {new_result['id']} 的结果 ({len(all_results)} 题总计)")
        
        return all_results

    # 记录性能统计总结
    def log_performance_summary(self, total_questions):
        if self.stats['start_time'] is None:
            self.logger.error("未调用 start_timing()，无法计算总时间")
            return
            
        total_time = time.time() - self.stats['start_time']
        
        # 记录到CSV文件
        self._save_performance_csv(total_questions, total_time)
        
        # 记录详细统计
        self.logger.info("="*50)
        self.logger.info("性能统计总结:")
        self.logger.info(f"  运行序号: {self.run_id}")
        self.logger.info(f"  模型名称: {self.model_name}")
        self.logger.info(f"  总题数: {total_questions}")
        self.logger.info(f"  总耗时: {total_time:.2f}s")
        self.logger.info(f"  平均耗时: {total_time/total_questions:.2f}s/题" if total_questions > 0 else "  平均耗时: N/A")
        
        for qtype, stats in self.stats['batch_times'].items():
            self.logger.info(f"  {qtype}: {stats['count']}题, 平均{stats['avg_time']:.2f}s/题")
        
        self.logger.info(f"  统计文件: {self.stats_file}")
        self.logger.info(f"  日志文件: {self.log_file}")
        self.logger.info("="*50)

    # 保存性能统计到CSV
    def _save_performance_csv(self, total_questions, total_time):
        # 准备CSV数据行
        row_data = {
            'run_id': self.run_id,
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_questions': total_questions,
            'total_time': round(total_time, 2),
            'avg_time_per_question': round(total_time / total_questions, 2) if total_questions > 0 else 0,
            'output_file': self.output_filename,
            'log_file': self.log_filename
        }
        
        # 添加各题型的统计数据
        all_types = ['choice', 'code-generate', 'generic-generate', 'math']
        for qtype in all_types:
            if qtype in self.stats['batch_times']:
                stats = self.stats['batch_times'][qtype]
                row_data[f'{qtype}_count'] = stats['count']
                row_data[f'{qtype}_total_time'] = round(stats['total_time'], 2)
                row_data[f'{qtype}_avg_time'] = round(stats['avg_time'], 2)
                row_data[f'{qtype}_generation_time'] = round(stats['generation_time'], 2)
            else:
                row_data[f'{qtype}_count'] = 0
                row_data[f'{qtype}_total_time'] = 0
                row_data[f'{qtype}_avg_time'] = 0
                row_data[f'{qtype}_generation_time'] = 0
        
        # 定义CSV表头
        fieldnames = [
            'run_id', 'model_name', 'timestamp', 'datetime', 
            'total_questions', 'total_time', 'avg_time_per_question',
            'output_file', 'log_file'
        ]
        for qtype in all_types:
            fieldnames.extend([
                f'{qtype}_count', f'{qtype}_total_time', 
                f'{qtype}_avg_time', f'{qtype}_generation_time'
            ])
        
        # 检查文件是否存在以及表头是否一致
        file_exists = os.path.exists(self.stats_file)
        need_header = True
        
        if file_exists:
            # 检查现有文件的表头是否与当前fieldnames一致
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    existing_reader = csv.DictReader(f)
                    existing_fieldnames = existing_reader.fieldnames
                    if existing_fieldnames == fieldnames:
                        need_header = False
                    else:
                        # 表头不一致，备份旧文件并重新创建
                        backup_file = self.stats_file + '.backup'
                        os.rename(self.stats_file, backup_file)
                        self.logger.info(f"表头不一致，已备份旧文件到: {backup_file}")
                        file_exists = False
            except (IOError, csv.Error):
                # 文件损坏或格式错误，重新创建
                if os.path.exists(self.stats_file):
                    backup_file = self.stats_file + '.backup'
                    os.rename(self.stats_file, backup_file)
                    self.logger.info(f"文件格式错误，已备份旧文件到: {backup_file}")
                file_exists = False
        
        # 写入CSV文件
        mode = 'a' if file_exists else 'w'
        with open(self.stats_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # 如果需要写入表头
            if need_header:
                writer.writeheader()
                if not file_exists:
                    self.logger.info(f"创建性能统计文件: {self.stats_file}")
                else:
                    self.logger.info(f"更新性能统计文件表头: {self.stats_file}")
            
            # 写入数据行
            writer.writerow(row_data)

    # 获取最新的结果文件路径，用于断点续传
    def get_latest_result_file(self):
        if not os.path.exists(self.model_logs_dir):
            return None
            
        result_files = [f for f in os.listdir(self.model_logs_dir) 
                       if f.endswith('.json') and f != self.output_filename]
        
        if not result_files:
            return None
        
        # 选择最新的结果文件
        latest_file = max(result_files, 
                         key=lambda x: os.path.getctime(os.path.join(self.model_logs_dir, x)))
        
        return os.path.join(self.model_logs_dir, latest_file)

    # 记录断点续传信息
    def log_recovery(self, recovery_file, recovered_count):
        self.logger.info(f"从 {recovery_file} 恢复历史进度，已处理 {recovered_count} 题")

    # 记录错误信息
    def log_error(self, error_message):
        self.logger.error(error_message)

    # 关闭日志器
    def close(self):
        # 清理handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler) 