import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from main import ActiveLearningPipeline
from experiments.ablation_config import ABLATION_SETTINGS
from utils.logger import logger


class ControlAblationRunner:
    def __init__(self, config, results_dir, run_id=None):
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.results_dir / "logs_md"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_results = {}
        self.start_time = datetime.now()
        self.run_id = run_id or self.start_time.strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.results_dir / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.config.RESULTS_DIR = str(self.results_dir)
        self.config.CHECKPOINT_DIR = os.path.join(self.config.RESULTS_DIR, "checkpoints")
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        
        if not getattr(self.config, "LLM_API_KEY", None):
            self.config.STOP_ON_LLM_FAILURE = False
            logger.info("LLM_API_KEY not set. Override: STOP_ON_LLM_FAILURE=False")
    
    def run_single_experiment(self, experiment_name):
        logger.info(f"\n{'='*80}")
        logger.info(f"CONTROL ABLATION | run_id={self.run_id} | experiment={experiment_name}")
        logger.info(f"{'='*80}")
        
        log_path = self.log_dir / f"{experiment_name}_{self.run_id}.md"
        
        try:
            pipeline = ActiveLearningPipeline(self.config, experiment_name, run_id=self.run_id)
            run_result = pipeline.run_and_collect(log_path=str(log_path))
            
            result = {
                'experiment_name': experiment_name,
                'description': ABLATION_SETTINGS[experiment_name]['description'],
                'control_permissions': ABLATION_SETTINGS[experiment_name].get('control_permissions', {}),
                'performance_history': run_result['performance_history'],
                'budget_history': run_result['budget_history'],
                'alc': float(run_result['alc']),
                'final_miou': float(run_result['final_miou']),
                'final_f1': float(run_result['final_f1']),
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'log_file': str(log_path)
            }
            
            self.all_results[experiment_name] = result
            logger.info(f"Experiment {experiment_name} completed. ALC: {result['alc']:.4f}, Final mIoU: {result['final_miou']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Experiment {experiment_name} failed: {str(e)}")
            
            result = {
                'experiment_name': experiment_name,
                'description': ABLATION_SETTINGS[experiment_name]['description'],
                'control_permissions': ABLATION_SETTINGS[experiment_name].get('control_permissions', {}),
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'log_file': str(log_path)
            }
            self.all_results[experiment_name] = result
            return result
    
    def run_control_ablation(self):
        control_experiments = [
            name for name in ABLATION_SETTINGS.keys()
            if name.startswith("agent_control_")
        ]
        
        logger.info(f"Starting control ablation experiments: {control_experiments}")
        
        for exp_name in control_experiments:
            self.run_single_experiment(exp_name)
        
        self.save_results()
        self.generate_report()
    
    def save_results(self):
        results_file = self.run_dir / "control_ablation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {results_file}")
    
    def generate_report(self):
        report_path = self.run_dir / "control_ablation_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 控制消融实验报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Run ID**: {self.run_id}\n\n")
            
            f.write("## 实验概述\n\n")
            f.write("本实验旨在验证Agent控制动作的必要性，通过限制Agent的控制权限来分析不同控制动作对性能的影响。\n\n")
            
            f.write("## 实验配置\n\n")
            f.write("| 实验名称 | 描述 | 允许的控制动作 |\n")
            f.write("|---------|------|----------------|\n")
            
            for exp_name, result in self.all_results.items():
                if result['status'] == 'success':
                    permissions = result.get('control_permissions', {})
                    allowed_actions = []
                    if permissions.get('set_lambda'):
                        allowed_actions.append('set_lambda')
                    if permissions.get('set_query_size'):
                        allowed_actions.append('set_query_size')
                    if permissions.get('set_epochs_per_round'):
                        allowed_actions.append('set_epochs_per_round')
                    f.write(f"| {exp_name} | {result['description']} | {', '.join(allowed_actions) if allowed_actions else '无'} |\n")
            
            f.write("\n## 实验结果\n\n")
            f.write("| 实验名称 | ALC | Final mIoU | Final F1 | 状态 |\n")
            f.write("|---------|-----|------------|----------|------|\n")
            
            for exp_name, result in self.all_results.items():
                if result['status'] == 'success':
                    f.write(f"| {exp_name} | {result['alc']:.4f} | {result['final_miou']:.4f} | {result['final_f1']:.4f} | {result['status']} |\n")
                else:
                    f.write(f"| {exp_name} | - | - | - | {result['status']} |\n")
            
            f.write("\n## 性能对比分析\n\n")
            
            successful_results = {k: v for k, v in self.all_results.items() if v['status'] == 'success'}
            
            if len(successful_results) >= 2:
                f.write("### ALC对比\n\n")
                alc_values = [(k, v['alc']) for k, v in successful_results.items()]
                alc_values.sort(key=lambda x: x[1], reverse=True)
                
                for exp_name, alc in alc_values:
                    f.write(f"- **{exp_name}**: {alc:.4f}\n")
                
                f.write("\n### mIoU对比\n\n")
                miou_values = [(k, v['final_miou']) for k, v in successful_results.items()]
                miou_values.sort(key=lambda x: x[1], reverse=True)
                
                for exp_name, miou in miou_values:
                    f.write(f"- **{exp_name}**: {miou:.4f}\n")
                
                f.write("\n### 控制动作增益分析\n\n")
                f.write("当前版本仅保留最小控制动作集合的对比，建议在多seed与统计检验后再下结论。\n\n")
            
            f.write("\n## 结论\n\n")
            f.write("通过控制消融实验，我们可以量化不同控制动作对Agent性能的贡献。")
            f.write("在该次运行中观察到不同控制动作对整体性能存在差异，后续需要多seed与统计检验以支持更强结论。\n\n")
        
        logger.info(f"Report generated at {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Run control ablation experiments')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Explicit run_id to use')
    parser.add_argument('--n_rounds', type=int, default=10,
                        help='Number of active learning rounds to run')
    
    args = parser.parse_args()
    
    config = Config()
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.Config()
    
    runner = ControlAblationRunner(config, args.results_dir, run_id=args.run_id)

    if args.n_rounds is not None and args.n_rounds > 0:
        config.N_ROUNDS = int(args.n_rounds)
        logger.info(f"Config override: N_ROUNDS={config.N_ROUNDS}")
    
    runner.run_control_ablation()


if __name__ == '__main__':
    main()
