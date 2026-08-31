def build_qa_prompt(
    question: str,
    contexts: list[dict],
    conversation_history: list[dict] | None = None,
) -> str:
    context_text = "\n\n".join(
        (
            f"[片段 {index} | 来源：{item.get('source', 'unknown')} | "
            f"页码：{format_page_range(item)} | 章节：{item.get('section', 'body')} | "
            f"章节标题：{item.get('section_title', 'unknown')} | "
            f"类型：{item.get('content_type', 'text')}"
            f"{format_caption(item)}]\n{item['text']}"
        )
        for index, item in enumerate(contexts, start=1)
    )
    history_text = build_history_text(conversation_history or [])
    source_names = sorted({str(item.get("source", "unknown")) for item in contexts})
    source_name_text = "、".join(source_names) if source_names else "unknown"

    return f"""请基于下面的论文片段回答问题。

要求：
1. 优先依据论文片段，不要凭空编造。
2. 如果片段中没有足够信息，请明确说明“当前资料不足”。
3. 回答要面向科研阅读，尽量指出算法、实验、创新点、局限或复现要点。
4. 可以用条目化方式回答。
5. 每个关键结论必须标注来源和页码，格式为“（来源：真实文件名，第 N 页）”。
6. 来源名称必须从本轮允许列表中逐字复制，不得使用 paper.pdf、论文.pdf 等示例名或自行改名。
   本轮允许的来源名称：{source_name_text}
7. 如果问题涉及多篇论文，请按论文分别回答，不要把不同论文的方法混在一起。
8. 不要根据参考文献列表、作者简介或致谢内容推断论文方法；这些片段只能说明资料不足。
9. 如果用户使用“它”“这篇”“上面的方法”等追问表达，可以参考最近对话理解指代；但事实依据仍必须来自本轮论文片段。
10. table 片段是从论文表格提取的 Markdown；比较指标时要同时核对表头、行名和数值。
11. figure 片段可能附带对应原图。如果本轮消息包含图片，请结合图像、Caption 和附近正文解读；如果没有图片，不要推断未明示的视觉细节。
12. 解读统计图时优先说明图类型、横纵坐标、图例、算法/数据系列、主要趋势和异常点；解读架构图时说明模块、连接关系、输入输出和数据流。
13. 对看不清的坐标值、图例文字或曲线细节，必须明确说明无法确认，不得猜测数值。

最近对话：
{history_text}

论文片段：
{context_text}

用户问题：
{question}
"""


def format_page_range(item: dict) -> str:
    start = item.get("page_start", item.get("page", "unknown"))
    end = item.get("page_end", start)
    return str(start) if start == end else f"{start}-{end}"


def format_caption(item: dict) -> str:
    caption = item.get("caption", "")
    return f" | 图表题：{caption}" if caption else ""


def build_history_text(conversation_history: list[dict], max_turns: int = 4) -> str:
    if not conversation_history:
        return "无"

    selected = conversation_history[-max_turns:]
    lines: list[str] = []
    for index, item in enumerate(selected, start=1):
        question = compact_text(item.get("question", ""), 220)
        answer = compact_text(item.get("answer", ""), 420)
        lines.append(f"[第 {index} 轮]\n用户：{question}\n助手：{answer}")
    return "\n\n".join(lines)


def compact_text(text: str, max_length: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[:max_length - 3]}..."
