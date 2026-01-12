import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import load_data
from model import Qwen3MoleculeLLM  # 你的自定义模型

# ===========================
# 1. collate function
# ===========================
def collate_fn(batch):
    """
    batch: list of dict
    input_ids, attention_mask, labels: list[int] -> tensor
    smiles: list[str] 保留
    """
    return {
        "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
        "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
        "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        "smiles": [str(x["smiles"]) for x in batch],  # 确保每个元素是 str
    }

# ===========================
# 2. 训练函数
# ===========================
def train_sft(model, tokenizer, train_dataset, epochs=1, batch_size=2, lr=1e-4, device="cuda"):
    model.train()
    model.to(device)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # ======== smiles 保证 list[str] ========
            smiles_list = [str(s) for s in batch["smiles"]]

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            print(smiles_list)

            # ===========================
            # forward
            # ===========================
            outputs = model(smiles_list, input_ids)  # 返回 logits
            logits = outputs

            # ======== text 部分 logits ========
            L_smiles = model.projector.num_queries + 2  # start + end + num_queries
            text_logits = logits[:, L_smiles:, :]

            # ======== causal shift loss ========
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # ===========================
            # backward & step
            # ===========================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

    return model

# ===========================
# 3. 初始化模型
# ===========================
model = Qwen3MoleculeLLM(qwen_model_name="/zengdaojian/zhangjia/BioLatent/Qwen4B")
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

# 加载 projector，如果有保存
try:
    model.projector.load_state_dict(torch.load("projector.pt"))
except FileNotFoundError:
    print("No saved projector found, using initialized projector.")

# ===========================
# 4. 构建训练数据
# ===========================
train_dataset = load_data("/zengdaojian/zhangjia/BioLatent/ChemCotDataset/chemcotbench-cot")

# ===========================
# 5. 训练
# ===========================
trained_model = train_sft(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    epochs=3,
    batch_size=2,
    lr=1e-4,
    device="cuda"
)

# ===========================
# 6. 保存模型
# ===========================
torch.save(trained_model.state_dict(), "qwen3_mol_sft_full.pt")
trained_model.llm.save_pretrained("./qwen3_mol_sft_llm")
tokenizer.save_pretrained("./qwen3_mol_sft_llm")
torch.save(trained_model.projector.state_dict(), "projector.pt")

# ===========================
# 7. 推理示例
# ===========================
trained_model.eval()

texts = ["Describe the functional groups of this molecule."]
smiles_list = ["CC(=O)OC1=CC=CC=C1C(=O)O"]
smiles_list = [str(s) for s in smiles_list]  # 确保是 str list

enc = tokenizer(texts, return_tensors="pt", padding=True)
text_ids = enc["input_ids"].cuda()

with torch.no_grad():
    logits = trained_model(smiles_list, text_ids)
    L_smiles = trained_model.projector.num_queries + 2
    text_logits = logits[:, L_smiles:, :]
    pred_ids = text_logits.argmax(dim=-1)
    pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

print("Generated text:", pred_text)
