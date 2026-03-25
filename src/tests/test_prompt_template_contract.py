import unittest

from src.agent.prompt_template import PromptBuilder


class TestPromptTemplateContract(unittest.TestCase):
    def test_system_prompt_contains_structured_blocks_and_constraints(self):
        builder = PromptBuilder()
        prompt = builder.build_system_prompt(
            total_iterations=16,
            current_iteration=8,
            last_miou=0.71,
            lambda_t=0.35,
            control_permissions={
                "get_system_status": True,
                "get_top_k_samples": True,
                "get_score_distribution": True,
                "finalize_selection": True,
                "set_lambda": False,
            },
        )
        self.assertIn("结构化输入（必须按块理解）", prompt)
        self.assertIn("diagnostics:", prompt)
        self.assertIn("issues:", prompt)
        self.assertIn("recent_history:", prompt)
        self.assertIn("Action.tool_name 必须属于动作空间", prompt)
        self.assertIn("不得输出越权请求", prompt)


if __name__ == "__main__":
    unittest.main()

