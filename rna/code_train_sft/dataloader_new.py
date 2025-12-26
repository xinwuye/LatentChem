from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer
from dataloader import load_data
from model_new import Qwen3MoleculeLLM  # 自定义的“分子 + LLM”融合模型
import torch


# =========================================================
# 自定义 collate_fn
# =========================================================
# 作用：
# 1. 对 batch 内样本进行 padding
# 2. 手动构造 labels，使其与模型输出 logits 对齐
# 3. 额外返回 smiles 字段（用于模型 forward 中的分子编码）
#
def collate_fn(
    batch,
    smiles_len=6,           # SMILES token 长度（4 个 SMILES + 2 个特殊 token）
    pad_token_id=0,         # input_ids 的 padding token
    label_pad_id=-100,      # labels 的 padding（CrossEntropyLoss 会忽略 -100）
):
    # 当前 batch 中最长的 input_ids 长度
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        ids = x["input_ids"]           # prompt + answer 的 token ids
        mask = x["attention_mask"]     # attention mask
        lab = x["labels"]              # 只包含 answer 部分的 labels

        pad_len = max_len - len(ids)

        # ===============================
        # input_ids / attention_mask
        # ===============================
        input_ids.append(
            ids + [pad_token_id] * pad_len
        )
        attention_mask.append(
            mask + [0] * pad_len
        )

        # ===============================
        # labels 构造（关键部分）
        # ===============================
        # logits 的顺序：
        # [SMILES tokens] + [prompt tokens] + [answer tokens]
        #
        # 但我们只希望在 answer tokens 上计算 loss
        # 因此：
        #   - SMILES + prompt → -100（忽略）
        #   - answer → lab
        #
        labels.append(
            [label_pad_id] * smiles_len +  # SMILES + special tokens，不参与 loss
            lab                             # 真实需要监督的答案 token
            +[0] * pad_len
            # 注意：这里不再额外 pad labels，
            # 因为 lab 本身已经与 input_ids 对齐
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),

        # 额外字段：SMILES 字符串
        # SFTTrainer 会保留该字段，并传入 model.forward
        "smiles": [x["smiles"].replace(".", "") for x in batch],
    }


# =========================================================
# 使用 TRL 的 SFTTrainer 进行 SFT 训练
# =========================================================
def train_sft_trl(
    model_name="/zengdaojian/zhangjia/BioLatent/Qwen4B",
    data_path="/zengdaojian/zhangjia/BioLatent/ChemCotDataset/chemcotbench-cot",
    output_dir="./qwen3_mol_sft_results",
    epochs=3,
    batch_size=2,
    lr=1e-4,
    device="cuda",
):
    # =====================================================
    # 初始化模型和 tokenizer
    # =====================================================
    # Qwen3MoleculeLLM 内部：
    #   - 包含 Qwen LLM
    #   - 包含 SMILES 编码器
    #   - 包含 projector（分子特征 → LLM embedding 空间）
    model = Qwen3MoleculeLLM(qwen_model_name=model_name)

    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token  # Qwen 通常用 eos 作为 pad

    # =====================================================
    # 加载已训练的 projector（如果存在）
    # =====================================================
    try:
        model.projector.load_state_dict(torch.load("projector.pt"))
    except FileNotFoundError:
        print("No saved projector found, using initialized projector.")

    # =====================================================
    # 加载训练数据
    # =====================================================
    # load_data 需要返回：
    # {
    #   "input_ids": ...
    #   "attention_mask": ...
    #   "labels": ...
    #   "smiles": ...
    # }
    train_dataset = load_data(data_path)

    # =====================================================
    # SFTConfig（等价于 TrainingArguments）
    # =====================================================
    training_args = SFTConfig(
        output_dir=output_dir,                  # 训练输出目录
        num_train_epochs=epochs,                # 训练轮数
        per_device_train_batch_size=batch_size, # 每张卡 batch size
        learning_rate=lr,                       # 学习率
        bf16=True if torch.cuda.is_available() else False,  # 是否使用 bf16
        max_seq_length=1024,                    # 最大序列长度
        packing=False,                          # 不做样本 packing
        dataset_text_field=None,                # 不使用默认 text 字段
        remove_unused_columns=False,             # 保留 smiles 等自定义字段
    )

    # =====================================================
    # 初始化 SFTTrainer
    # =====================================================
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,  # ⭐ 使用你自定义的 collate_fn
    )

    # =====================================================
    # 开始训练
    # =====================================================
    trainer.train()

    # =====================================================
    # 保存模型
    # =====================================================
    # 保存 LLM（包含 LoRA / projector 之外的部分）
    trainer.save_model("./qwen3_mol_sft_llm")

    # 单独保存 projector（因为是你自定义模块）
    torch.save(model.projector.state_dict(), "projector.pt")

    return trainer.model


# =========================================================
# 示例入口
# =========================================================
if __name__ == "__main__":
    trained_model = train_sft_trl()
