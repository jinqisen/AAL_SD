import json
from .config import AgentThresholds, AgentConstraints

class PromptBuilder:
    def build_system_prompt(self, total_iterations=0, current_iteration=0, last_miou=0.0, lambda_t=0.0, rollback_threshold=None, rollback_mode=None, k_definition=None, control_permissions=None, require_explicit_lambda: bool = False, miou_low_gain_streak=0):
        allowed_actions = []
        control_permissions_json = "null"
        alpha_allowed = None
        
        # 默认权限设置 (Getter 默认 True, Setter 默认 False)
        # 注意：这里需要与 Toolbox 保持一致，但 PromptBuilder 是无状态的，所以依赖传入的 control_permissions
        # 如果 control_permissions 为 None，假设全开(Getter)或全关(Setter)? 
        # 稳妥起见，我们解析传入的 dict，如果没传则不做假设（显示所有 standard tools）
        
        effective_permissions = {
            "get_system_status": True,
            "get_top_k_samples": True,
            "get_sample_details": True,
            "set_hyperparameter": False,
            "finalize_selection": True,
            "set_lambda": False,
            "set_query_size": False,
            "set_epochs_per_round": False,
            "get_score_distribution": True
        }
        
        if isinstance(control_permissions, dict):
            control_permissions_json = json.dumps(control_permissions, ensure_ascii=False)
            effective_permissions.update(control_permissions)
            if effective_permissions.get("set_lambda"):
                allowed_actions.append("set_lambda")
            if effective_permissions.get("set_query_size"):
                allowed_actions.append("set_query_size")
            if effective_permissions.get("set_epochs_per_round"):
                allowed_actions.append("set_epochs_per_round")
            alpha_allowed = bool(effective_permissions.get("set_alpha", False))
            if alpha_allowed:
                allowed_actions.append("set_hyperparameter")
        
        allowed_actions_str = ", ".join(allowed_actions) if allowed_actions else "无"
        action_space = list(allowed_actions) if allowed_actions else []
        if "finalize_selection" not in action_space:
            action_space.append("finalize_selection")
        action_space_str = "{" + ", ".join(action_space) + "}"
        alpha_allowed_str = "未知"
        if alpha_allowed is True:
            alpha_allowed_str = "是"
        if alpha_allowed is False:
            alpha_allowed_str = "否"
        require_explicit_lambda_str = "是" if bool(require_explicit_lambda) else "否"
        explicit_lambda_note = ""
        if bool(require_explicit_lambda) and bool(effective_permissions.get("set_lambda")):
            explicit_lambda_note = "- 本实验要求显式 set_lambda：在调用 get_top_k_samples 前必须先 set_lambda（否则会失败）。"

        # 动态构建工具列表
        all_tools_definitions = [
            ("get_system_status", "1. get_system_status()\n            - 返回系统状态信息"),
            ("get_top_k_samples", "2. get_top_k_samples(k, lambda_param)\n            - 返回前K个候选样本"),
            ("get_sample_details", "3. get_sample_details(sample_id)\n            - 返回样本详细信息"),
            ("set_hyperparameter", "4. set_hyperparameter(alpha)\n            - 设置超参数alpha"),
            ("finalize_selection", "5. finalize_selection(sample_ids, reason)\n            - 提交最终选择"),
            ("set_lambda", "6. set_lambda(lambda_value, scope=\"round\")\n            - 设置融合权重lambda(0-1)"),
            ("set_query_size", "7. set_query_size(query_size, scope=\"round\")\n            - 设置本轮标注数量，自动裁剪到剩余预算"),
            ("set_epochs_per_round", "8. set_epochs_per_round(epochs, scope=\"round\")\n            - 设置每轮训练epoch数，上限20"),
            ("get_score_distribution", "9. get_score_distribution(n_bins, quantiles)\n            - 返回U/K分布统计与直方图")
        ]
        
        available_tools_list = []
        for key, desc in all_tools_definitions:
            # 特殊处理：set_alpha 对应 tool 名 set_hyperparameter
            perm_key = "set_alpha" if key == "set_hyperparameter" else key
            if effective_permissions.get(perm_key, False):
                available_tools_list.append(desc)
        
        available_tools_str = "\n            ".join(available_tools_list)

        rollback_threshold_val = None
        try:
            rollback_threshold_val = float(rollback_threshold) if rollback_threshold is not None else None
        except Exception:
            rollback_threshold_val = None
        if rollback_threshold_val is None:
            rollback_threshold_val = float(AgentThresholds.ROLLBACK_THRESHOLD)
        rollback_mode_str = str(rollback_mode) if rollback_mode is not None else "adaptive_threshold"

        kd = str(k_definition or "coreset_to_labeled").strip()
        if kd == "coreset_to_labeled_fixed":
            k_definition_desc = (
                "1. coreset-to-labeled-fixed：K(x)=min_{l∈L0}||f_x-f_l|| 的归一化（L0为初始标注池，固定不随轮次扩张；越大越新颖/覆盖不足）。"
                "若初始标注为空：视为实验配置/数据错误，必须终止运行并修复初始标注池。"
            )
        else:
            k_definition_desc = (
                "1. coreset-to-labeled：K(x)=min_{l∈L}||f_x-f_l|| 的归一化（L为当前标注池；越大越新颖/覆盖不足）。"
                "若已标注为空：视为实验配置/数据错误，必须终止运行并修复初始标注池。"
            )

        prompt = """
            你是一个主动学习系统的“动态控制器”（不是聊天助手），你的输出必须直接驱动本轮Round的控制动作与最终选样。
            总体目标：在固定总标注预算下最大化学习曲线面积(ALC)与最终mIoU，并降低不稳定性（回撤、过拟合、训练过载、预算越界、无效动作）。

            情境感知:
            - 总迭代轮数 (T_max): {total_iterations}
            - 当前迭代轮数 (t): {current_iteration}
            - 当前模型性能 (mIoU): {last_miou:.4f}
            - 当前连续低增益轮数: {miou_low_gain_streak}
            - 当前自适应权重 (lambda_t): {lambda_t:.4f}
            - 是否要求显式设置lambda: {require_explicit_lambda_str}
            - 控制权限(JSON): {control_permissions_json}
            - 允许控制动作: {allowed_actions_str}
            - set_hyperparameter 可用: {alpha_allowed_str}

            AD-KUCS 评分逻辑（必须遵守） ：
            - 不确定性 U(x) = entropy(p_x)，p_x为样本x预测概率分布。
            - 知识增益 K(x) 定义：
              {k_definition_desc}
            - 融合得分 Score(x) = (1-λ)·U(x) + λ·K(x)，λ∈[0,1]。
            - 系统提供的 lambda_t 为本轮策略建议/已应用的λ；建议将单轮调整幅度控制在±{lambda_adjust_range}以内。系统对λ的硬约束为[{lambda_min},{lambda_max}]；实验策略额外约束为 λ∈[{lambda_policy_min},{lambda_policy_max}]。

            量化阈值（必须遵守）：
            - 显著为负：miou_delta < {significant_negative}
            - 后期：t / T_max > {late_stage_ratio}
            - 回撤阈值：miou_delta < {rollback_threshold}
            - 过拟合警戒：grad_train_val_cos_last < {overfit_tvc_warn}
            - 过拟合严重：grad_train_val_cos_last < {overfit_tvc_severe}
            - 训练过载：epochs > {training_overload_epochs} 且 miou_delta < 0
            - 无效动作：调用 set_* 但参数未发生变化

            控制与评价机制（在Thought里显式说明）：
            - 观测维度：训练趋势、U/K分布统计、预算边界。
            - 动作空间：{action_space_str}。
            - 约束：remaining_budget硬约束；epochs<={epochs_cap}；λ∈[{lambda_min},{lambda_max}]；触发裁剪必须在reason记录。
            - 选择策略：
            - U分布整体高（均值>{high_uncertainty_mean}）且方差大（std>{high_uncertainty_std}）：降低λ偏向U。
            - 进入后期或K分布尾部更长（75分位>均值+{long_tail_k_threshold}）：提高λ偏向K。
            - rollback_flag为真（表示本轮mIoU出现显著回撤）或miou_delta显著为负：不得提高λ，必要时小幅降低λ，并标注"回撤-保守控制"。
            - 若观测到过拟合严重：以实验策略为准（可能采用 severe_logic=and/or、risk_trigger=ci/abs、severe_tvc_key=grad_train_val_cos_last/min）。触发后下一轮降低λ（例如 λ=max(λ-{lambda_delta_down},{lambda_policy_min})），且不得提高λ。
            - 若观测到过拟合风险低（overfit_risk<={overfit_risk_lo}）且 miou_gain 连续低：才允许小幅上调λ（例如 λ=min(λ+{lambda_delta_up},{lambda_policy_max})），并将单轮上调控制在步长限制内。
            - 若连续低增益轮数(miou_low_gain_streak) >= 3 且无明显过拟合：说明当前样本冗余度高，应增加探索（提高λ，例如+0.05~0.1），以引入更多新颖样本打破停滞。

            异常处理（必须遵守）：
            - miou为NaN/Inf：使用上轮有效值并注明“miou异常-使用历史值”。
            - 分布统计为空：跳过分布分析，使用lambda_t并注明“分布数据缺失-使用默认λ”。
            - 工具调用失败：重试1次，仍失败则使用默认值并注明“工具调用失败-使用默认值”。
            - 参数非法：使用默认值（λ=0.5 / epochs=10 / query_size=remaining_budget）并注明原因。

            可用工具：
            {available_tools_str}

            执行顺序（必须遵循）：
            1) 观测：使用 get_system_status 与 get_score_distribution 获取状态与分布（每种观测工具每轮最多调用2次；通常1次足够）。
            2) 控参：根据约束调用 set_lambda 等可用参数设置工具（若有权限）。
            {explicit_lambda_note}
            3) 选样：使用 get_top_k_samples 与 get_sample_details。
            4) 提交：使用 finalize_selection 提交样本与reason（必须在总步数上限内完成，否则该轮视为失败）。
            约束：建议每阶段最多调用{max_tool_calls_per_phase}次工具；系统同时限制总步数不超过{max_steps}步。

            ReAct格式（仅输出Thought与Action）：
            Thought: <包含观测摘要、动作三元组与约束考虑、预期收益与风险>
            Action: {{"tool_name": "...", "parameters": {{...}}}}

            严格约束：
            - 只能输出两行：Thought 与 Action（不要输出 Observation，不要使用代码块）。
            - Action 行必须以 "Action:" 开头，后面紧跟单个 JSON 对象，且 JSON 后不要追加任何文字。
            """
        formatted = prompt.format(
            total_iterations=int(total_iterations or 0),
            current_iteration=int(current_iteration or 0),
            last_miou=float(last_miou or 0.0),
            miou_low_gain_streak=int(miou_low_gain_streak or 0),
            lambda_t=float(lambda_t or 0.0),
            control_permissions_json=control_permissions_json,
            allowed_actions_str=allowed_actions_str,
            action_space_str=action_space_str,
            alpha_allowed_str=alpha_allowed_str,
            significant_negative=AgentThresholds.SIGNIFICANT_NEGATIVE,
            late_stage_ratio=AgentThresholds.LATE_STAGE_RATIO,
            rollback_threshold=float(rollback_threshold_val),
            rollback_mode=rollback_mode_str,
            k_definition_desc=k_definition_desc,
            training_overload_epochs=AgentThresholds.TRAINING_OVERLOAD_EPOCHS,
            epochs_cap=AgentThresholds.EPOCHS_CAP,
            lambda_min=AgentConstraints.LAMBDA_MIN,
            lambda_max=AgentConstraints.LAMBDA_MAX,
            lambda_policy_min=getattr(AgentThresholds, "LAMBDA_CLAMP_MIN", AgentConstraints.LAMBDA_MIN),
            lambda_policy_max=getattr(AgentThresholds, "LAMBDA_CLAMP_MAX", AgentConstraints.LAMBDA_MAX),
            lambda_delta_down=getattr(AgentThresholds, "LAMBDA_DELTA_DOWN", 0.1),
            lambda_delta_up=getattr(AgentThresholds, "LAMBDA_DELTA_UP", 0.05),
            overfit_risk_hi=getattr(AgentThresholds, "OVERFIT_RISK_HI", 0.8),
            overfit_risk_lo=getattr(AgentThresholds, "OVERFIT_RISK_LO", 0.2),
            overfit_tvc_min_hi=getattr(AgentThresholds, "OVERFIT_TVC_MIN_HI", 0.5),
            high_uncertainty_mean=AgentThresholds.HIGH_UNCERTAINTY_MEAN,
            high_uncertainty_std=AgentThresholds.HIGH_UNCERTAINTY_STD,
            long_tail_k_threshold=AgentThresholds.LONG_TAIL_K_THRESHOLD,
            min_labeled_for_novelty=AgentThresholds.MIN_LABELED_FOR_NOVELTY,
            lambda_adjust_range=AgentThresholds.LAMBDA_ADJUST_RANGE,
            overfit_tvc_warn=AgentThresholds.OVERFIT_TVC_WARN,
            overfit_tvc_severe=AgentThresholds.OVERFIT_TVC_SEVERE,
            max_tool_calls_per_phase=int(getattr(AgentConstraints, "MAX_TOOL_CALLS_PER_PHASE", 2) or 2),
            max_steps=int(getattr(AgentConstraints, "MAX_STEPS", 10) or 10),
            available_tools_str=available_tools_str,
            require_explicit_lambda_str=require_explicit_lambda_str,
            explicit_lambda_note=explicit_lambda_note,
        )
        return formatted.replace('{{', '{').replace('}}', '}')

    def build_user_prompt(self, labeled_size=None, unlabeled_size=None, query_size=None):
        if query_size is None:
            base = "新一轮主动学习开始。请使用 get_system_status 获取系统状态并依据lambda_t选择最优样本。"
        else:
            base = f"新一轮主动学习开始。请使用 get_system_status 获取系统状态并依据lambda_t选择最优样本（如无错误，不要重复调用 get_system_status）。需选择{int(query_size)}个样本。"
        context = {}
        if labeled_size is not None:
            context["labeled_pool_size"] = int(labeled_size)
        if unlabeled_size is not None:
            context["unlabeled_pool_size"] = int(unlabeled_size)
        if context:
            return base + "\n\n上下文:\n" + json.dumps(context, ensure_ascii=False)
        return base
