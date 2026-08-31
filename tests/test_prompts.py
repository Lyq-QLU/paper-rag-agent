from src.prompts import build_qa_prompt


def test_prompt_requires_real_source_name():
    prompt = build_qa_prompt(
        "使用了什么模型？",
        [{"text": "evidence", "source": "01_家庭医疗.pdf", "page": 3}],
    )
    assert "本轮允许的来源名称：01_家庭医疗.pdf" in prompt
    assert "例如“（来源：paper.pdf" not in prompt
