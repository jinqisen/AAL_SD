import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class ReportGenerator:
    def __init__(self, results: Dict, output_dir: Path):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._manifest_cache = None

    def _load_manifest(self) -> Dict:
        if isinstance(self._manifest_cache, dict):
            return self._manifest_cache
        manifest_path = self.output_dir / "manifest.json"
        if not manifest_path.exists():
            self._manifest_cache = {}
            return self._manifest_cache
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._manifest_cache = payload if isinstance(payload, dict) else {}
        except Exception:
            self._manifest_cache = {}
        return self._manifest_cache

    def _manifest_config(self) -> Dict:
        manifest = self._load_manifest()
        cfg = manifest.get("config") if isinstance(manifest, dict) else None
        return cfg if isinstance(cfg, dict) else {}

    def _fmt_percent(self, value) -> str:
        try:
            v = float(value)
        except Exception:
            return "N/A"
        return f"{v * 100:.1f}%"

    def _trend_word(self, delta: float) -> str:
        try:
            d = float(delta)
        except Exception:
            return "变化"
        if d > 0:
            return "提升"
        if d < 0:
            return "下降"
        return "无明显变化"
    
    def generate_all_reports(self):
        self.generate_summary_report()
        self.generate_baseline_comparison_report()
        self.generate_ablation_study_report()
        self.generate_detailed_results_report()
    
    def generate_summary_report(self):
        cfg = self._manifest_config()
        initial_ratio = self._fmt_percent(cfg.get("INITIAL_LABELED_SIZE"))
        test_ratio = self._fmt_percent(cfg.get("TEST_SIZE"))
        n_rounds = cfg.get("N_ROUNDS") or self._get_n_rounds()
        query_size = cfg.get("QUERY_SIZE") or self._get_query_size()
        total_budget = cfg.get("TOTAL_BUDGET")

        report = f"""# AAL-SD 实验结果摘要报告

**生成时间**: {self.timestamp}

---

## 1. 实验概览

本报告总结了AAL-SD框架在Landslide4Sense数据集上的基线对比和消融实验结果。

### 1.1 实验配置

| 配置项 | 值 |
|-------|-----|
| 数据集 | Landslide4Sense |
| 初始标注比例 | {initial_ratio} |
| 测试集比例 | {test_ratio} |
| 主动学习轮数 | {n_rounds} |
| 每轮查询样本数 | {query_size} |
| 总标注预算 | {total_budget if total_budget is not None else 'N/A'} |

### 1.2 实验列表

| 实验名称 | 描述 | 状态 |
|---------|------|------|
"""
        for exp_name, result in self.results.items():
            status_icon = "✅" if result.get('status') == 'success' else "❌"
            report += f"| {exp_name} | {result.get('description', 'N/A')} | {status_icon} |\n"
        
        report += "\n---\n"
        
        successful_results = {k: v for k, v in self.results.items() if v.get('status') == 'success'}
        
        if successful_results:
            report += "## 2. 核心结果对比\n\n"
            report += "### 2.1 性能指标汇总\n\n"
            report += "| 实验名称 | ALC | 最终 mIoU | 最终 F1-Score |\n"
            report += "|---------|-----|----------|--------------|\n"
            
            for exp_name, result in successful_results.items():
                alc = result.get('alc', 0)
                miou = result.get('final_miou', 0)
                f1 = result.get('final_f1', 0)
                report += f"| {exp_name} | {alc:.4f} | {miou:.4f} | {f1:.4f} |\n"
            
            report += "\n### 2.2 性能排名\n\n"
            
            sorted_by_alc = sorted(successful_results.items(), key=lambda x: x[1].get('alc', 0), reverse=True)
            report += "**按ALC排名**:\n\n"
            for i, (exp_name, result) in enumerate(sorted_by_alc, 1):
                report += f"{i}. {exp_name}: {result.get('alc', 0):.4f}\n"
            
            report += "\n"
            sorted_by_miou = sorted(successful_results.items(), key=lambda x: x[1].get('final_miou', 0), reverse=True)
            report += "**按最终mIoU排名**:\n\n"
            for i, (exp_name, result) in enumerate(sorted_by_miou, 1):
                report += f"{i}. {exp_name}: {result.get('final_miou', 0):.4f}\n"

            report += "\n### 2.3 topk回退统计\n\n"
            report += "| 实验名称 | 回退轮次数 | 回退补齐总数 | 回退比例 |\n"
            report += "|---------|-----------|------------|---------|\n"
            for exp_name, result in successful_results.items():
                fallback_summary = self._summarize_fallback(result.get('fallback_history', []))
                report += f"| {exp_name} | {fallback_summary['fallback_rounds']} | {fallback_summary['fallback_added']} | {fallback_summary['fallback_ratio']:.2%} |\n"
        
        report += "\n---\n"
        report += f"\n*报告生成于 {self.timestamp}*\n"
        
        self._save_report('summary_report.md', report)
    
    def generate_baseline_comparison_report(self):
        report = f"""# 基线方法对比报告

**生成时间**: {self.timestamp}

---

## 1. 基线方法说明

本研究对比了以下基线方法：

| 基线方法 | 类别 | 策略描述 |
|---------|------|---------|
| Random Sampling | 无策略 | 随机选择样本进行标注，作为最低性能参照 |
| Entropy Sampling | 经典不确定性 | 选择模型输出平均像素熵最高的样本 |
| Core-Set | 经典多样性 | 选择能最大程度覆盖整个未标注池特征空间的样本子集 |
| BALD | 混合策略SOTA | 贝叶斯主动学习，同时考虑不确定性和模型分歧 |
| LLM-US | LLM对照组 | LLM仅基于不确定性分数进行决策 |
| LLM-RS | LLM对照组 | LLM仅基于随机分数进行决策 |

---

## 2. 基线对比结果

### 2.1 性能-成本曲线对比

| 实验名称 | ALC | 最终 mIoU | 最终 F1-Score |
|---------|-----|----------|--------------|
"""
        baseline_experiments = ['baseline_random', 'baseline_entropy', 'baseline_coreset', 
                               'baseline_bald', 'baseline_llm_us', 'baseline_llm_rs']
        
        for exp_name in baseline_experiments:
            if exp_name in self.results and self.results[exp_name].get('status') == 'success':
                result = self.results[exp_name]
                report += f"| {exp_name} | {result.get('alc', 0):.4f} | {result.get('final_miou', 0):.4f} | {result.get('final_f1', 0):.4f} |\n"
        
        report += "\n### 2.2 详细性能对比\n\n"
        
        successful_baselines = {k: v for k, v in self.results.items() 
                               if k in baseline_experiments and v.get('status') == 'success'}
        
        if successful_baselines:
            report += "#### 2.2.1 每轮mIoU变化\n\n"
            report += "| 轮次 | " + " | ".join(successful_baselines.keys()) + " |\n"
            report += "|------|" + "|".join(["--------"] * len(successful_baselines)) + "|\n"
            
            n_rounds = self._get_n_rounds()
            for round_idx in range(n_rounds):
                row = [str(round_idx + 1)]
                for exp_name in successful_baselines:
                    perf_history = successful_baselines[exp_name].get('performance_history', [])
                    if round_idx < len(perf_history):
                        row.append(f"{perf_history[round_idx]['mIoU']:.4f}")
                    else:
                        row.append("N/A")
                report += "| " + " | ".join(row) + " |\n"
            
            report += "\n#### 2.2.2 标注效率分析\n\n"
            report += "达到不同mIoU水平所需的标注样本数：\n\n"
            report += "| 目标 mIoU | " + " | ".join(successful_baselines.keys()) + " |\n"
            report += "|----------|" + "|".join(["--------"] * len(successful_baselines)) + "|\n"
            
            target_mious = [0.5, 0.6, 0.7, 0.75]
            for target in target_mious:
                row = [f"{target:.2f}"]
                for exp_name in successful_baselines:
                    perf_history = successful_baselines[exp_name].get('performance_history', [])
                    budget = self._get_budget_for_target(perf_history, target)
                    row.append(str(budget) if budget is not None else "未达到")
                report += "| " + " | ".join(row) + " |\n"

            report += "\n### 2.3 预算—性能曲线数据\n\n"
            for exp_name in successful_baselines:
                report += f"#### {exp_name}\n\n"
                report += self._format_curve_table(successful_baselines[exp_name].get('performance_history', []))
                report += "\n"
        
        report += "\n---\n"
        report += f"\n*报告生成于 {self.timestamp}*\n"
        
        self._save_report('baseline_comparison_report.md', report)
    
    def generate_ablation_study_report(self):
        full_model_name = "full_model_A_lambda_policy" if ("full_model_A_lambda_policy" in self.results) else "full_model"
        report = f"""# 消融实验报告

**生成时间**: {self.timestamp}

---

## 1. 消融实验设计

为验证AD-KUCS算法及AAL-SD框架中每个创新组件的必要性，设计了以下消融实验：

| 实验设置 | 描述 | 验证目标 |
|---------|------|---------|
| {full_model_name} | 完整模型（λ: policy闭环；Agent不允许 set_lambda） | 作为性能上限 |
| no_agent | 移除Agent；λ由sigmoid自适应(随标注进度变化) | 验证LLM控制λ的增益 |
| uncertainty_only | 固定λ=0，仅使用不确定性 | 验证知识增益模块的必要性 |
| knowledge_only | 固定λ=1，仅使用知识增益 | 验证不确定性模块的必要性 |
| fixed_lambda | 固定λ=0.5 | 验证λ_t动态自适应调整的有效性 |

---

## 2. 消融实验结果

### 2.1 性能对比

| 实验名称 | ALC | 最终 mIoU | 最终 F1-Score |
|---------|-----|----------|--------------|
"""
        ablation_experiments = [full_model_name, 'no_agent',
                               'uncertainty_only', 'knowledge_only', 'fixed_lambda']
        
        for exp_name in ablation_experiments:
            if exp_name in self.results and self.results[exp_name].get('status') == 'success':
                result = self.results[exp_name]
                report += f"| {exp_name} | {result.get('alc', 0):.4f} | {result.get('final_miou', 0):.4f} | {result.get('final_f1', 0):.4f} |\n"
        
        report += "\n### 2.2 组件有效性分析\n\n"
        
        successful_ablations = {k: v for k, v in self.results.items() 
                               if k in ablation_experiments and v.get('status') == 'success'}
        
        if full_model_name in successful_ablations:
            full_alc = successful_ablations[full_model_name].get('alc', 0)
            full_miou = successful_ablations[full_model_name].get('final_miou', 0)
            
            report += "#### 2.2.1 Agent组件贡献\n\n"
            if 'no_agent' in successful_ablations:
                no_agent_alc = successful_ablations['no_agent'].get('alc', 0)
                no_agent_miou = successful_ablations['no_agent'].get('final_miou', 0)
                alc_gain = ((full_alc - no_agent_alc) / no_agent_alc * 100) if no_agent_alc > 0 else 0
                miou_gain = ((full_miou - no_agent_miou) / no_agent_miou * 100) if no_agent_miou > 0 else 0
                report += f"- **ALC提升**: {alc_gain:+.2f}% ({no_agent_alc:.4f} → {full_alc:.4f})\n"
                report += f"- **mIoU提升**: {miou_gain:+.2f}% ({no_agent_miou:.4f} → {full_miou:.4f})\n"
                report += f"- **结论**: 在该次运行中观察到性能{self._trend_word(alc_gain)}\n"
            
            report += "\n#### 2.2.2 知识增益模块贡献\n\n"
            if 'uncertainty_only' in successful_ablations:
                u_only_alc = successful_ablations['uncertainty_only'].get('alc', 0)
                u_only_miou = successful_ablations['uncertainty_only'].get('final_miou', 0)
                alc_gain = ((full_alc - u_only_alc) / u_only_alc * 100) if u_only_alc > 0 else 0
                miou_gain = ((full_miou - u_only_miou) / u_only_miou * 100) if u_only_miou > 0 else 0
                report += f"- **ALC提升**: {alc_gain:+.2f}% ({u_only_alc:.4f} → {full_alc:.4f})\n"
                report += f"- **mIoU提升**: {miou_gain:+.2f}% ({u_only_miou:.4f} → {full_miou:.4f})\n"
                report += f"- **结论**: 在该次运行中观察到性能{self._trend_word(alc_gain)}\n"
            
            report += "\n#### 2.2.3 不确定性模块贡献\n\n"
            if 'knowledge_only' in successful_ablations:
                k_only_alc = successful_ablations['knowledge_only'].get('alc', 0)
                k_only_miou = successful_ablations['knowledge_only'].get('final_miou', 0)
                alc_gain = ((full_alc - k_only_alc) / k_only_alc * 100) if k_only_alc > 0 else 0
                miou_gain = ((full_miou - k_only_miou) / k_only_miou * 100) if k_only_miou > 0 else 0
                report += f"- **ALC提升**: {alc_gain:+.2f}% ({k_only_alc:.4f} → {full_alc:.4f})\n"
                report += f"- **mIoU提升**: {miou_gain:+.2f}% ({k_only_miou:.4f} → {full_miou:.4f})\n"
                report += f"- **结论**: 在该次运行中观察到性能{self._trend_word(alc_gain)}\n"
            
            report += "\n#### 2.2.4 动态权重贡献\n\n"
            if 'fixed_lambda' in successful_ablations:
                fixed_alc = successful_ablations['fixed_lambda'].get('alc', 0)
                fixed_miou = successful_ablations['fixed_lambda'].get('final_miou', 0)
                alc_gain = ((full_alc - fixed_alc) / fixed_alc * 100) if fixed_alc > 0 else 0
                miou_gain = ((full_miou - fixed_miou) / fixed_miou * 100) if fixed_miou > 0 else 0
                report += f"- **ALC提升**: {alc_gain:+.2f}% ({fixed_alc:.4f} → {full_alc:.4f})\n"
                report += f"- **mIoU提升**: {miou_gain:+.2f}% ({fixed_miou:.4f} → {full_miou:.4f})\n"
                report += f"- **结论**: 在该次运行中观察到性能{self._trend_word(alc_gain)}\n"
        
        report += "\n### 2.3 每轮性能变化\n\n"
        
        if successful_ablations:
            report += "| 轮次 | " + " | ".join(successful_ablations.keys()) + " |\n"
            report += "|------|" + "|".join(["--------"] * len(successful_ablations)) + "|\n"
            
            n_rounds = self._get_n_rounds()
            for round_idx in range(n_rounds):
                row = [str(round_idx + 1)]
                for exp_name in successful_ablations:
                    perf_history = successful_ablations[exp_name].get('performance_history', [])
                    if round_idx < len(perf_history):
                        row.append(f"{perf_history[round_idx]['mIoU']:.4f}")
                    else:
                        row.append("N/A")
                report += "| " + " | ".join(row) + " |\n"

            report += "\n### 2.4 预算—性能曲线数据\n\n"
            for exp_name in successful_ablations:
                report += f"#### {exp_name}\n\n"
                report += self._format_curve_table(successful_ablations[exp_name].get('performance_history', []))
                report += "\n"
        
        report += "\n---\n"
        report += f"\n*报告生成于 {self.timestamp}*\n"
        
        self._save_report('ablation_study_report.md', report)
    
    def generate_detailed_results_report(self):
        report = f"""# 详细实验结果报告

**生成时间**: {self.timestamp}

---

## 实验详细结果

"""
        for exp_name, result in self.results.items():
            report += f"### {exp_name}\n\n"
            report += f"**描述**: {result.get('description', 'N/A')}\n\n"
            report += f"**状态**: {result.get('status', 'unknown')}\n\n"
            
            if result.get('status') == 'success':
                report += f"**ALC**: {result.get('alc', 0):.4f}\n\n"
                report += f"**最终 mIoU**: {result.get('final_miou', 0):.4f}\n\n"
                report += f"**最终 F1-Score**: {result.get('final_f1', 0):.4f}\n\n"
                fallback_history = result.get('fallback_history', [])
                if fallback_history:
                    fallback_summary = self._summarize_fallback(fallback_history)
                    report += "**topk回退统计**:\n\n"
                    report += f"- 回退轮次数: {fallback_summary['fallback_rounds']}\n"
                    report += f"- 回退补齐总数: {fallback_summary['fallback_added']}\n"
                    report += f"- 回退比例: {fallback_summary['fallback_ratio']:.2%}\n\n"
                    report += "| 轮次 | 回退 | 保留样本数 | 回退补齐数 |\n"
                    report += "|------|------|-----------|-----------|\n"
                    for item in fallback_history:
                        report += f"| {item.get('round')} | {'是' if item.get('fallback_used') else '否'} | {item.get('preferred_count', 0)} | {item.get('fallback_added', 0)} |\n"
                    report += "\n"
                
                report += "#### 性能历史\n\n"
                report += "| 轮次 | mIoU | F1-Score | 标注样本数 |\n"
                report += "|------|------|----------|-----------|\n"
                
                for perf in result.get('performance_history', []):
                    report += f"| {perf['round']} | {perf['mIoU']:.4f} | {perf['f1_score']:.4f} | {perf['labeled_size']} |\n"
                
                report += "\n"
            else:
                report += f"**错误**: {result.get('error', 'Unknown error')}\n\n"
            
            report += "---\n\n"
        
        report += f"\n*报告生成于 {self.timestamp}*\n"
        
        self._save_report('detailed_results_report.md', report)
    
    def _save_report(self, filename: str, content: str):
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Report saved to {filepath}")
    
    def _get_n_rounds(self) -> int:
        for result in self.results.values():
            if result.get('status') == 'success' and 'performance_history' in result:
                return len(result['performance_history'])
        return 3
    
    def _get_query_size(self) -> int:
        for result in self.results.values():
            if result.get('status') == 'success' and 'performance_history' in result:
                history = result['performance_history']
                if len(history) >= 2:
                    return history[1]['labeled_size'] - history[0]['labeled_size']
        return 100
    
    def _get_budget_for_target(self, perf_history: List, target_miou: float) -> int:
        for perf in perf_history:
            if perf['mIoU'] >= target_miou:
                return perf['labeled_size']
        return None

    def _format_curve_table(self, perf_history: List) -> str:
        if not perf_history:
            return "未生成曲线数据。\n"
        lines = []
        lines.append("| 轮次 | 标注样本数 | mIoU | F1-Score |")
        lines.append("|------|-----------|------|----------|")
        for perf in perf_history:
            lines.append(f"| {perf['round']} | {perf['labeled_size']} | {perf['mIoU']:.4f} | {perf['f1_score']:.4f} |")
        return "\n".join(lines) + "\n"

    def _summarize_fallback(self, fallback_history: List) -> Dict:
        if not fallback_history:
            return {
                "fallback_rounds": 0,
                "fallback_added": 0,
                "fallback_ratio": 0.0
            }
        total_rounds = len(fallback_history)
        fallback_rounds = sum(1 for item in fallback_history if item.get("fallback_used"))
        fallback_added = sum(int(item.get("fallback_added", 0) or 0) for item in fallback_history)
        fallback_ratio = fallback_rounds / total_rounds if total_rounds else 0.0
        return {
            "fallback_rounds": fallback_rounds,
            "fallback_added": fallback_added,
            "fallback_ratio": fallback_ratio
        }
