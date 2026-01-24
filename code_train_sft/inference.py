import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    TrainerCallback, 
    DataCollatorForSeq2Seq
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import logging
from datetime import datetime
import argparse
import inspect
import wandb
import plotext as plt
import json

# 导入我们的自定义组件
from model_stage3 import Qwen3MoleculeLLM
from dataloader import load_data, COCONUT_TOKENS
from config import ModelConfig
# from train_sft_stage2 import MultiModalDataCollator, MultiModalSFTTrainer, LoraTrainingMonitorCallback, TerminalPlotCallback
import torch.nn.functional as F
import random

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: extract this logic into model_stage3.py
from train_stage3 import load_trained_components_stage3

def load_test_data(test_data_path, include_tasks, max_len=None):
    """
    load test data dispatcher
    exclude subtasks per-bench here
    """

    if max_len is None:
        max_len = ModelConfig.MAX_TEXT_LEN

    logger.info(f"Loading test/eval data from: {test_data_path} (eval_mode=True)")

    if "ChemCoTBench" in test_data_path:
        dataset = load_data(test_data_path, include_cot=False, is_coconut=False, eval_mode=True, include_tasks=include_tasks, exclude_tasks=['gsk', 'jnk','logp','solubility','sol','qed'], max_len=max_len)
        logger.info(f"Loaded tokenized eval dataset ChemCoTBench from dir: {len(dataset)} examples")
    elif "ChemCoTDataset" in test_data_path:
        dataset = load_data(test_data_path, include_cot=False, is_coconut=False, eval_mode=True, include_tasks=include_tasks, exclude_tasks=['rcr'], max_len=max_len)
        logger.info(f"Loaded tokenized eval dataset ChemCoTBench from dir: {len(dataset)} examples")
    else:
    
        dataset = load_data(test_data_path, include_cot=False, is_coconut=False, eval_mode=True, max_len=max_len)#exclude_tasks=['molecule_design'],
        logger.info(f"Loaded tokenized eval dataset ChemCoTBench from dir: {len(dataset)} examples")
    
    return dataset

def prepare_evaluation_dataset(
    test_data_path,
    tokenization_max_len=None,
    max_samples=None,
    include_tasks = None,
    proc_index: int = 0,
    num_procs: int = 1,
    ):
    """
    
    """
    logger.info(f"Preparing evaluation metadata from {test_data_path}")
    dataset = load_test_data(test_data_path, include_tasks, max_len=tokenization_max_len)
    original_total = len(dataset)

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))


    # Per-process strided slicing (if requested)
    if num_procs > 1:
        total = len(dataset)
        indices = list(range(proc_index, total, num_procs))
        dataset = dataset.select(indices)
        logger.info(
            f"Process {proc_index}/{num_procs} -> assigned {len(dataset)} samples (original total {original_total})"
        )

    metadata = {
        "original_total": original_total,
        "final_count": len(dataset),
        "proc_index": proc_index,
        "num_procs": num_procs,
    }

    # Build per-sample metadata aligned with `dataset` ordering.
    per_sample_metadata = []
    for local_idx in range(len(dataset)):
        item = dataset[local_idx]
        # Keep the original smiles (not cleaned) as part of metadata
        smiles_list = item.get("smiles", [])
        sample_meta = {
            "sample_id": local_idx * num_procs + proc_index,
            "smiles": smiles_list,
            "task": item.get("task", None),
        }
        per_sample_metadata.append(sample_meta)

    logger.info(f"Number of eval samples to run: {len(dataset)}")
    return dataset, metadata, per_sample_metadata

def run_inference_on_dataset(
    model,
    tokenizer,
    dataset,
    device,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    inference_batch_size: int = 8,
    num_return_sequences: int = 1,
):
    """Run inference over a prepared dataset using mini-batches.

    Returns:
        generation_outputs: list of dicts, each containing at least 'result' (str or None)
                            and optionally 'error' when an exception occurred.
    """
    logger.info(f"Running inference (device={device}, max_new_tokens={max_new_tokens}, batch_size={inference_batch_size})")

    model.eval()
    model = model.to(device)

    generation_outputs = []

    n = len(dataset)
    if n == 0:
        return generation_outputs

    # Determine pad token id fallback
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    # Process in mini-batches
    for start_idx in tqdm(range(0, n, inference_batch_size)):
        end_idx = min(start_idx + inference_batch_size, n)
        batch_indices = list(range(start_idx, end_idx))
        # Collect per-sample items
        input_tensors = []
        smiles_batch = []
        raw_items = []
    
        # TODO: batch inputs in dataloader
        for idx in batch_indices:
            item = dataset[idx]
            raw_items.append(item)
            smiles_list = item.get("smiles", [])
            cleaned_smiles = [s.replace(".", "").strip() for s in smiles_list]
            smiles_batch.append(cleaned_smiles)

            input_ids_seq = torch.tensor(item["input_ids"], dtype=torch.long)
            input_tensors.append(input_ids_seq)

            # ignore original attention masks, as our tokenizers don't do padding
            # TODO: support original attention masks
            
        lengths = torch.tensor([len(s) for s in input_tensors])
        padded_input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=pad_token_id).to(device)
        max_len = padded_input_ids.size(1)
        padded_attention_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        padded_attention_mask = padded_attention_mask.to(device).long()
        
        with torch.inference_mode():
            generated_ids = model.generate(
                smiles_list=smiles_batch,
                input_ids=padded_input_ids,
                attention_mask=padded_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                num_return_sequences=num_return_sequences,
            ) # (B * N, L)
        
        try:
            del padded_input_ids
            del padded_attention_mask
            del input_tensors
            del lengths
        except:
            pass
        
        generated_ids = generated_ids.cpu()
        if isinstance(generated_ids, torch.Tensor):
            gen_tensor = generated_ids
        else:
            gen_tensor = torch.stack(generated_ids) if isinstance(generated_ids, (list, tuple)) else torch.tensor(generated_ids)
        
        batch_size = len(batch_indices)
        if gen_tensor.size(0) != batch_size * num_return_sequences:
            raise Exception(f"Generated tensor batch size ({gen_tensor.size(0)}) != expected ({batch_size} * {num_return_sequences}).")
        
        gen_tensor_cpu = gen_tensor.tolist()
        decoded_list = tokenizer.batch_decode(gen_tensor_cpu, skip_special_tokens=True)
        if num_return_sequences > 1:
            for i in range(batch_size):
                results = decoded_list[i * num_return_sequences : (i + 1) * num_return_sequences]
                generation_outputs.append({"results": [result.strip() for result in results]})
        else:     
            for decoded in decoded_list:
                generation_outputs.append({"result": decoded.strip()})
        
        try: 
            del gen_tensor
            del gen_tensor_cpu
            del generated_ids
        except:
            pass

    return generation_outputs

def save_inference_results(
    save_results_path,
    per_sample_metadata,
    generation_outputs,
    model,
    test_data_path,
    generation_config,
    device,
    ):
    '''
    per-process response + metadata -> save to json
    all results shards will be merged and dispatched to tasks in the eval pipeline
    '''
    if len(per_sample_metadata) != len(generation_outputs):
        logger.warning("Per-sample metadata and generation outputs length mismatch when saving results.")

    merged_results = []
    for i in range(min(len(per_sample_metadata), len(generation_outputs))):
        merged = dict(per_sample_metadata[i]) # copy metadata
        # merge generation-side fields (result, error if present)
        merged.update(generation_outputs[i])
        merged_results.append(merged)

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "test_data_path": test_data_path,
        "model_info": {
            "device": str(device),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "generation_config": generation_config,
        "num_samples": len(merged_results),
        "test_results": merged_results,
    }

    os.makedirs(os.path.dirname(save_results_path) if os.path.dirname(save_results_path) else ".", exist_ok=True)
    with open(save_results_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {save_results_path}")

def inference_stage3():
    parser = argparse.ArgumentParser(description="Stage 3 Training for Bio-LatentCOT")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None, help="Stage 2 LoRA weights (optional)")
    parser.add_argument("--projector_path", type=str, default=None, help="Unified projector + bio_updater weights (optional)")
    parser.add_argument("--inference_results_path", type=str, default="./outputs/stage3_coconut")
    parser.add_argument("--c_thought", type=int, default=2, help="Number of latent tokens per CoT step")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--training_stage", type=int, default=3, choices=[1, 2, 3], help="Which stage to train: 1 (No COT), 2 (With COT), 3 (Latent/Coconut)")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="生成的最大token数（用于inference模式）")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度（用于inference模式）")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样参数（用于inference模式）")
    parser.add_argument("--max_test_samples", type=int, default=None, help="最大测试样本数，None表示全部测试（用于inference模式）")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="生成重复次数（用于inference模式）")
    parser.add_argument("--include_tasks", type=str, nargs="*", default=None, help="需要包含的任务类型，None表示全部任务（用于inference模式）")
    # Stage 3 switches (only effective when --training_stage 3)
    # parser.add_argument(
    #     "--is_coconut",
    #     type=lambda x: (str(x).lower() == "true"),
    #     default=True,
    #     help="Whether to run Coconut latent training for stage 3 (ignored for stage 1/2).",
    # )
    parser.add_argument(
        "--is_both_latent",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable Bio-latent thinker tokens for stage 3 (ignored for stage 1/2).",
    )
    parser.add_argument(
        "--is_biothinker",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable BioThinker (bio-latent block) when --is_both_latent is false.",
    )
    parser.add_argument(
        "--is_biothinker_multi",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Use BioThinkerMulti (4-expert weighted FFN) instead of BioThinker.",
    )
    parser.add_argument(
        "--is_taskthinker",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable TaskThinker (task-latent block) when --is_both_latent is false.",
    )
    parser.add_argument(
        "--is_taskthinker_multi",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Use TaskThinkerMulti (4-expert weighted MLP) instead of TaskThinker.",
    )
    parser.add_argument(
        "--is_bioupdater",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable BioUpdater (memory update) when --is_both_latent is false.",
    )
    parser.add_argument(
        "--is_bioupdater_multi",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Use BioTokenUpdaterMulti (4-expert weighted FFN) instead of BioTokenUpdater.",
    )
    parser.add_argument(
        "--is_bioupdater_gating",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable BioUpdater gating (Linear+Sigmoid hard switch). When false, behavior is unchanged.",
    )
    parser.add_argument(
        "--is_biothinker_gating",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable BioThinker gating (hard switch). When gate=0, bio-latent block is replaced by anchor embedding.",
    )
    parser.add_argument(
        "--is_taskthinker_gating",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable TaskThinker gating (hard switch). Gate scales the MLP residual: x + gate*y.",
    )
    parser.add_argument(
        "--bio_latent_lambda",
        type=float,
        default=0.0,
        help="Weight for bio-latent cosine hinge loss (only effective when --training_stage 3 and --is_both_latent true).",
    )
    parser.add_argument(
        "--bio_latent_alpha",
        type=float,
        default=0.5,
        help="Margin alpha for bio-latent cosine hinge loss: mean(max(0, alpha - cos(v, mu))).",
    )
    parser.add_argument(
        "--max_cot_string_len",
        type=int,
        default=2048,
        help="Max CoT string length used to scale task-latent count: ceil(len(cot)/max_cot_string_len*4).",
    )
    parser.add_argument(
        "--task_latent_max_steps",
        type=int,
        default=10,
        help="Max loop steps when generating task latents during inference (get_prompt_embeddings).",
    )
    
    parser.add_argument("--proc_index", type=int, default=0,
                    help="当前进程索引 (0-based)，用于样本分片")
    parser.add_argument("--num_procs", type=int, default=1,
                        help="并行进程总数（样本分片数）")
    parser.add_argument("--gpu", type=int, default=None,
                        help="显卡 id，优先于 proc_index (如果提供则使用此 GPU)")
    
    args = parser.parse_args()
    
    # 1. 基础配置
    mol_config = {
        'num_queries': ModelConfig.NUM_QUERIES,
        'input_dim': ModelConfig.INPUT_DIM,
        'num_heads': ModelConfig.NUM_HEADS
    }
    
    # 当前的权重路径，初始为参数传入的路径
    current_lora_path = args.lora_path
    current_projector_path = args.projector_path
    
    if args.training_stage == 1:
        stages = [0]
        is_coconut = False
        is_both_latent = False
        is_biothinker = bool(args.is_biothinker)
        is_taskthinker = bool(args.is_taskthinker)
        is_bioupdater = bool(args.is_bioupdater)
        is_biothinker_multi = bool(args.is_biothinker_multi)
        is_taskthinker_multi = bool(args.is_taskthinker_multi)
        is_bioupdater_multi = bool(args.is_bioupdater_multi)
        is_bioupdater_gating = bool(args.is_bioupdater_gating)
        is_biothinker_gating = bool(args.is_biothinker_gating)
        is_taskthinker_gating = bool(args.is_taskthinker_gating)
        bio_latent_lambda = 0.0
        bio_latent_alpha = 0.5
        max_cot_string_len = 2048
        task_latent_max_steps = 10
        mode_name = "Stage1-NoCOT"
    elif args.training_stage == 2:
        stages = [0]
        is_coconut = False
        is_both_latent = False
        is_biothinker = bool(args.is_biothinker)
        is_taskthinker = bool(args.is_taskthinker)
        is_bioupdater = bool(args.is_bioupdater)
        is_biothinker_multi = bool(args.is_biothinker_multi)
        is_taskthinker_multi = bool(args.is_taskthinker_multi)
        is_bioupdater_multi = bool(args.is_bioupdater_multi)
        is_bioupdater_gating = bool(args.is_bioupdater_gating)
        is_biothinker_gating = bool(args.is_biothinker_gating)
        is_taskthinker_gating = bool(args.is_taskthinker_gating)
        bio_latent_lambda = 0.0
        bio_latent_alpha = 0.5
        max_cot_string_len = 2048
        task_latent_max_steps = 10
        mode_name = "Stage2-WithCOT"
    else: # Stage 3
        is_coconut = False
        is_both_latent = bool(args.is_both_latent)
        is_biothinker = bool(args.is_biothinker)
        is_taskthinker = bool(args.is_taskthinker)
        is_bioupdater = bool(args.is_bioupdater)
        is_biothinker_multi = bool(args.is_biothinker_multi)
        is_taskthinker_multi = bool(args.is_taskthinker_multi)
        is_bioupdater_multi = bool(args.is_bioupdater_multi)
        is_bioupdater_gating = bool(args.is_bioupdater_gating)
        is_biothinker_gating = bool(args.is_biothinker_gating)
        is_taskthinker_gating = bool(args.is_taskthinker_gating)
        bio_latent_lambda = float(args.bio_latent_lambda)
        bio_latent_alpha = float(args.bio_latent_alpha)
        max_cot_string_len = int(args.max_cot_string_len)
        task_latent_max_steps = int(args.task_latent_max_steps)
        if is_coconut:
            stages = [0]
            mode_name = "Stage3-Coconut"
        else:
            stages = [0]
            mode_name = "Stage3-WithCOT"
            
    for stage in stages:
        logger.info(f"\n" + "🚀" * 30)
        logger.info(f"STARTING {mode_name} (STAGE {stage})")
        if is_coconut:
            logger.info(f"Replace first {stage} steps with {stage * args.c_thought} latents")
        if is_both_latent:
            logger.info("Bio-latent thinker enabled (N_bio_latents = #smiles).")
        logger.info("🚀" * 30 + "\n")

        # 2.1 初始化模型
        model = Qwen3MoleculeLLM(
            qwen_model_name=ModelConfig.DEFAULT_QWEN_PATH,
            mol_config=mol_config,
            is_coconut=is_coconut,
            is_both_latent=is_both_latent,
            is_biothinker=is_biothinker,
            is_taskthinker=is_taskthinker,
            is_bioupdater=is_bioupdater,
            is_biothinker_multi=is_biothinker_multi,
            is_taskthinker_multi=is_taskthinker_multi,
            is_bioupdater_multi=is_bioupdater_multi,
            is_bioupdater_gating=is_bioupdater_gating,
            is_biothinker_gating=is_biothinker_gating,
            is_taskthinker_gating=is_taskthinker_gating,
            bio_latent_lambda=bio_latent_lambda,
            bio_latent_alpha=bio_latent_alpha,
            max_cot_string_len=max_cot_string_len,
            task_latent_max_steps=task_latent_max_steps,
        )
        tokenizer = model.tokenizer
        
        # 加载 ckpt 的权重
        if current_lora_path or current_projector_path:
            logger.info(f"Loading weights for inference...")
            model = load_trained_components_stage3(
                model, 
                lora_weights_path=current_lora_path, 
                mm_projector_path=current_projector_path
            )
        
        # 确保 LoRA 已配置
        if not hasattr(model.model, 'peft_config') or model.model.peft_config is None:
            raise ValueError("LoRA weights not found. Please ensure that the model has been trained with LoRA.")
        
        # 确保投影器和 Bio Updater 冻结
        for param in model.projector.parameters():
            param.requires_grad = False
        
        for param in model.bio_updater.parameters():
            param.requires_grad = False
        if getattr(model, "bio_updater_gate", None) is not None:
            for param in model.bio_updater_gate.parameters():
                param.requires_grad = False

        if hasattr(model, "bio_thinker"):
            for param in model.bio_thinker.parameters():
                param.requires_grad = False
        if getattr(model, "bio_thinker_gate", None) is not None:
            for param in model.bio_thinker_gate.parameters():
                param.requires_grad = False
                
        if hasattr(model, "task_thinker"):
            for param in model.task_thinker.parameters():
                param.requires_grad = False
        if getattr(model, "task_thinker_gate", None) is not None:
            for param in model.task_thinker_gate.parameters():
                param.requires_grad = False
        
        model.model.eval()

        # 2.2 load test set and metadata
        dataset, metadata, per_sample_metadata = prepare_evaluation_dataset(
            test_data_path=args.data_path,
            tokenization_max_len=min(args.max_seq_length, ModelConfig.MAX_TEXT_LEN),
            max_samples=args.max_test_samples,
            include_tasks=args.include_tasks,
            proc_index=args.proc_index,
            num_procs=args.num_procs,
        )
        
        # 2.3 batched inference
        results = run_inference_on_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            inference_batch_size=args.batch_size,
            num_return_sequences=args.num_return_sequences,
        )
        
        # 3. save results
        
        save_inference_results(
            save_results_path=args.inference_results_path,
            per_sample_metadata=per_sample_metadata,
            generation_outputs=results,
            model=model,
            test_data_path=args.data_path,
            generation_config={
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p
            },
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        del model
        torch.cuda.empty_cache()

    logger.info(f"🎉 All {mode_name} inference Stages completed!")
    
if __name__ == "__main__":
    inference_stage3()