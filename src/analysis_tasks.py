from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisTask:
    name: str
    prompt: str


ANALYSIS_TASKS: list[AnalysisTask] = [
    AnalysisTask(
        name="结构化总结",
        prompt=(
            "请基于论文内容进行结构化总结。按以下字段回答："
            "1. 研究问题；2. 核心方法；3. 创新点；4. 算法流程；"
            "5. 实验数据集或实验场景；6. 对比算法；7. 评价指标；"
            "8. 局限性；9. 适合复现的部分。每条都要引用来源页码。"
        ),
    ),
    AnalysisTask(
        name="提取核心方法",
        prompt=(
            "请提取每篇论文采用的核心方法。若有多篇论文，请按论文分别回答。"
            "重点说明算法名称、输入、输出、关键步骤、是否使用强化学习/启发式/神经网络/进化算法。"
            "不要依据参考文献或作者简介推断，必须引用来源页码。"
        ),
    ),
    AnalysisTask(
        name="总结创新点",
        prompt=(
            "请总结论文的创新点。按论文分别列出：已有方法的问题、本文提出的改进、"
            "创新点对应的方法模块或实验结果。每个结论都要引用来源页码。"
        ),
    ),
    AnalysisTask(
        name="分析实验设置",
        prompt=(
            "请分析论文的实验设置。提取实验数据集、问题规模、对比算法、评价指标、"
            "消融实验和主要实验结论。若资料不足，请明确指出缺失哪一部分。"
        ),
    ),
    AnalysisTask(
        name="对比算法判断",
        prompt=(
            "请判断这些论文中的方法是否适合作为我的对比算法。请从研究问题、目标函数、"
            "约束条件、输入输出、算法类型、实验场景、可复现难度七个角度分析，"
            "最后给出适合/部分适合/不适合的结论，并引用来源页码。"
        ),
    ),
    AnalysisTask(
        name="复现建议",
        prompt=(
            "请给出论文复现建议。包括：复现目标、核心模块拆解、输入输出格式、"
            "需要实现的算法步骤、实验指标、可能的难点、最小可复现实验版本。"
            "所有建议必须基于论文片段，不足之处要说明。"
        ),
    ),
    AnalysisTask(
        name="多论文对比",
        prompt=(
            "请对多篇论文进行对比分析。按表格思路组织回答：论文名称、研究问题、"
            "核心方法、算法类型、实验场景、对比算法、优点、局限、和我的路径优化/调度研究的关系。"
            "每篇论文至少引用一个来源页码。"
        ),
    ),
]


def get_task_names() -> list[str]:
    return [task.name for task in ANALYSIS_TASKS]


def get_task_prompt(name: str) -> str:
    for task in ANALYSIS_TASKS:
        if task.name == name:
            return task.prompt
    return ""

