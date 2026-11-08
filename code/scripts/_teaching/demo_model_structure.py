"""加载 Qwen3 模型，打印结构信息，跑一次推理演示。

用法：
    # 使用默认模型（Qwen3-1.7B）和默认提示词
    uv run python code/scripts/demo_model_structure.py

    # 指定模型路径
    uv run python code/scripts/demo_model_structure.py --model-path model_files/Qwen3-4B

    # 指定提示词
    uv run python code/scripts/demo_model_structure.py --prompt "什么是机器学习？"
"""

from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch


# 默认停用词：运行时查词表换成 token id，生成到该 token 就立即停止
#   <|im_end|>      —— Qwen3 对话模板的回合结束标记（normal 流程下它就是 eos，这里显式写出来便于讲解）
#   <|endoftext|>   —— Qwen 系列文本结束标记
#   "\n"            —— 换行符在词表里是单个 token，可演示“生成到换行就停”
DEFAULT_STOP_WORDS = ["<|im_end|>", "<|endoftext|>", "\n"]


def resolve_stop_token_ids(tokenizer, stop_words: list[str]) -> list[int]:
    """把停用词查词表，转成对应的 token id。

    只有「在词表里恰好是 1 个 token」的词才能做 token 级截止；
    像「再见」这种会被切成多个 token 的词，这里跳过并提示。

    Args:
        tokenizer: 分词器对象
        stop_words: 停用词列表（词表里的字面形式）

    Returns:
        list[int]: 停用词对应的 token id 列表
    """
    stop_ids, skipped = [], []
    for word in stop_words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) == 1:
            stop_ids.append(token_ids[0])
        else:
            skipped.append(f"{word!r}(被切成 {len(token_ids)} 个 token)")

    if skipped:
        print(f"  ⚠ 以下停用词不是单个 token，已跳过：{'、'.join(skipped)}")
    return stop_ids


class StopWordCriteria(StoppingCriteria):
    """词表停用词截止：新生成的最后一个 token 命中停用词，就停止生成。

    StoppingCriteria 是 transformers 的「提前停止」钩子，每生成一个 token 调用一次，
    返回 True 表示本序列生成结束。
    """

    def __init__(self, stop_token_ids: list[int]):
        # 转成 tensor 放在 CPU，调用时再对齐到 input_ids 所在设备
        self.stop_token_ids = torch.tensor(stop_token_ids, dtype=torch.long)

    def __call__(self, input_ids, scores, **kwargs) -> torch.Tensor:
        # input_ids 形状为 [batch_size, seq_len]，只看最后一个 token
        last_tokens = input_ids[:, -1]
        return torch.isin(last_tokens, self.stop_token_ids.to(last_tokens.device))


def load_model(model_path: Path):
    """加载模型和分词器。

    Args:
        model_path: 模型目录路径，应包含 config.json 和*.safetensors 文件

    Returns:
        tuple: (model, tokenizer) 模型和分词器对象
    """
    print(f"正在加载模型：{model_path}")

    # 检查模型目录是否存在
    if not model_path.exists():
        raise FileNotFoundError(
            f"模型目录不存在：{model_path}\n"
            f"请先运行：uv run python code/scripts/99_download_model.py"
        )

    # 加载分词器
    print("  加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 加载模型（用 bfloat16 省显存）
    print("  加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",  # 自动分配到可用设备（GPU/CPU）
    )

    print(f"  ✓ 模型加载完成")
    return model, tokenizer


def print_model_structure(model):
    """打印模型结构信息。

    Args:
        model: 已加载的模型对象
    """
    print("\n" + "="*60)
    print("模型结构信息")
    print("="*60)

    # 基本参数
    config = model.config
    print(f"模型类型：{config.model_type}")
    print(f"词元表大小：{config.vocab_size:,}")
    print(f"隐藏层维度：{config.hidden_size:,}")
    print(f"注意力头数：{config.num_attention_heads}")
    print(f"KV 头数：{config.num_key_value_heads}")
    print(f"层数：{config.num_hidden_layers}")
    print(f"最大上下文长度：{config.max_position_embeddings:,}")

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量：{total_params:,} ({total_params/1e9:.2f}B)")
    print(f"可训练参数：{trainable_params:,}")

    # 层结构：打印每个参数的完整名称、形状与参数量
    layers = model.model.layers
    print(f"\n共 {len(layers)} 个 Transformer 层，各层结构完全相同。")
    print(f"以第 0 层为例，层内参数如下：")
    print(f"  {'参数名':<38} {'形状':<18} {'参数量':>12}")
    print(f"  {'-'*38} {'-'*18} {'-'*12}")
    for name, param in layers[0].named_parameters():
        shape = str(tuple(param.shape))
        print(f"  {name:<38} {shape:<18} {param.numel():>12,}")

    # 层外的参数（词嵌入、最终 LayerNorm、lm_head）
    print(f"\n层外参数：")
    print(f"  {'参数名':<38} {'形状':<18} {'参数量':>12}")
    print(f"  {'-'*38} {'-'*18} {'-'*12}")
    for name, param in model.named_parameters():
        if ".layers." not in name:
            shape = str(tuple(param.shape))
            print(f"  {name:<38} {shape:<18} {param.numel():>12,}")


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    stop_words: list[str] | None = None,
):
    """跑一次推理演示，用「长度上限」和「词表停用词」两种方式控制生成截止。

    Args:
        model: 已加载的模型对象
        tokenizer: 分词器对象
        prompt: 输入提示词
        max_new_tokens: 最多生成的新 token 数（长度截止）
        stop_words: 停用词列表，命中即停止；None 用 DEFAULT_STOP_WORDS，[] 表示关闭停用词截止
    """
    if stop_words is None:
        stop_words = DEFAULT_STOP_WORDS
    stop_token_ids = resolve_stop_token_ids(tokenizer, stop_words)

    print("\n" + "="*60)
    print("推理演示")
    print("="*60)
    print(f"输入：{prompt}")
    print(f"停用词：{stop_words} → token id {stop_token_ids}")
    print(f"截止条件：最多 {max_new_tokens} 个新 token" + ("，或生成到停用词" if stop_token_ids else "（未启用停用词）"))

    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # 生成（贪心解码，保证结果可复现）
    print("\n正在生成...")
    stopping_criteria = (
        StoppingCriteriaList([StopWordCriteria(stop_token_ids)])
        if stop_token_ids
        else None
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,   # ① 长度截止
        stopping_criteria=stopping_criteria,  # ② 停用词截止：每生成一个 token 查一次词表
        do_sample=False,  # 贪心解码
        temperature=None,
    )

    # 只解码新生成的部分，避免把提示词原样再打印一遍
    new_ids = outputs[0][input_len:]
    # skip_special_tokens=True 会滤掉 <|im_end|> 这类特殊 token，换行符则靠 strip() 去掉
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # 说明到底是哪个条件截止的，方便课堂对照
    last_token_id = outputs[0][-1].item()
    if last_token_id in stop_token_ids:
        reason = f"命中停用词 token：{last_token_id} → {tokenizer.decode([last_token_id])!r}"
    elif len(new_ids) >= max_new_tokens:
        reason = "达到长度上限 max_new_tokens"
    else:
        reason = f"模型自己生成了结束标记（eos, id={last_token_id}）"

    print(f"\n实际生成 {len(new_ids)} 个 token（上限 {max_new_tokens}）")
    print(f"截止原因：{reason}")
    print(f"输出：{response}")


def main():
    """主函数：加载 → 打印结构 → 推理。"""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="【必填】你机器上模型目录的绝对路径。例如：/你的路径/环境和物料/models/models/Qwen--Qwen3-1.7B/snapshots/master/"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好，请简单介绍一下自己。",
        help="推理提示词（默认：你好，请简单介绍一下自己。）"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="最多生成的新 token 数（默认：100）"
    )
    parser.add_argument(
        "--stop-words",
        type=str,
        default=None,
        help=(
            "停用词，多个用逗号分隔，生成到该 token 即停止。"
            "换行写 \\n，例如：'<|im_end|>,\\n'。"
            "不传则用内置默认值；传空字符串表示关闭停用词截止。"
        )
    )
    args = parser.parse_args()

    # 停用词：None 用默认值，空字符串则关闭
    stop_words = None
    if args.stop_words is not None:
        stop_words = [
            w.replace("\\n", "\n") for w in args.stop_words.split(",") if w
        ]

    # 1. 加载模型
    model, tokenizer = load_model(args.model_path)

    # 2. 打印结构
    print_model_structure(model)

    # 3. 跑推理
    run_inference(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        stop_words=stop_words,
    )


if __name__ == "__main__":
    main()
