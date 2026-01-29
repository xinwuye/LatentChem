import os
import json
import logging
import statistics
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
import pandas as pd
import regex as re

from core.utils import extract_answer
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

answer_pattern = re.compile(r'<answer\s*>(.*?)</answer\s*>', flags=re.S)

tokenizer = AutoTokenizer.from_pretrained("/BioLatent/Bio-LatentCOT/models/Qwen3-8B-Base")

class BaseTaskEvaluator(ABC):
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def extract_answer(self, pred: str, task: str) -> str:
        return extract_answer(pred)
    
    @abstractmethod
    def extract_gt(self, gt_raw_item: Any, task: str) -> Any:
        """从单个 gt_raw[i] 提取 gt（可覆盖）。"""
        raise NotImplementedError

    @abstractmethod
    def evaluate_predictions(self, preds: List[List[str]], gts: List[Any], total_len: int, metadata: Any = None, task_name: str = None) -> Dict[str, float]:
        """对给定 preds/gts 计算评价指标（可覆盖）。
        返回字典形式的指标，例如 {"bleu": 0.8, ...}
        """
        raise NotImplementedError
    
    @abstractmethod
    def prepare_metadata(self, sample: Dict[str, Any]) -> Any:
        """从单个 sample 准备 metadata（可覆盖）。"""
        raise NotImplementedError
    
    def _load_samples(self, log_dir: str, model_name: str) -> List[dict]:
        path = os.path.join(log_dir, f"{model_name}.json")
        if not os.path.exists(path):
            raise ValueError(f"log file not found: {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def _load_gt_raw(self, gt_path: str, taskname: str) -> List[Any]:
        path = os.path.join(gt_path, taskname + '.json')
        if not os.path.exists(path):
            raise ValueError(f"gt file not found: {path}")
        with open(path, 'r') as f:
            return json.load(f)
    
    
    def _cot_length(self, result:str) -> int:
        matches = list(answer_pattern.finditer(result))
        if matches:
            cut_pos = matches[-1].start()
            trimmed_result = result[:cut_pos]
            text = trimmed_result
        else:
            text = result
            
        out = tokenizer(
            text,
            add_special_tokens=True,   # 是否计入 [CLS]/[SEP]
            return_length=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        
        return out['length'][0]
    
    def evaluate_score(
        self,
        model_name: str,
        sample_count: int,
        gt_path: str,
        logs_dir: str,
        task_name: str,
    ) -> Dict[str, Any]:
        log_dir = os.path.join(logs_dir, task_name)
        if not os.path.exists(log_dir):
            self.logger.warning(f"log not found: {log_dir}")
            return {}

        samples = self._load_samples(log_dir, model_name)
        gt_raw = self._load_gt_raw(gt_path, task_name)
        
        if len(samples) != len(gt_raw):
            raise ValueError(f"sample count {len(samples)} does not match gt count {len(gt_raw)}")

        preds: List[List[str]] = [[] for _ in range(sample_count)]
        metadata: List[Dict[str, Any]] = []
        gts: List[Any] = []
        
        invalid_num = 0
        
        for i, sample in enumerate(samples):
            
            # 支持 sample['result'] 或 sample['results'] 两种格式
            if 'result' in sample:
                if sample_count > 1:
                    raise ValueError("sample_count should be 1 when result is in sample")
                pred = self.extract_answer(sample['result'], task_name)
                # this is confusing:
                # None, [], 0.0 or "" are all seen as invalid predictions
                # But only None will be skipped
                # This is to be consistent with some benchmarks that calculate scores on every results regardless of their validity
                # However, valid-rate is used to indicate whether the model is following instructions
                # By default, extract_answer returns None for any invalid prediction
                # If you want to change this behavior, you can override extract_answer
                if not pred:
                    invalid_num += 1
                if pred is None:
                    continue
                preds[0].append(pred)

            elif 'results' in sample:
                if sample_count == 1:
                    raise ValueError("sample_count should be greater than 1 when results is in sample")
                results = sample['results']
                if isinstance(results, str):
                    results = json.loads(results)
                if sample_count != len(results):
                    invalid_num += 1
                    continue
                for j in range(sample_count):
                    pred = self.extract_answer(results[j], task_name)
                    if pred is None:
                        pred = ""
                        invalid_num += 1
                    preds[j].append(pred)

            else:
                raise ValueError("sample should have either 'result' or 'results'")
            
            gts.append(self.extract_gt(gt_raw[i], task_name))
            metadata.append(self.prepare_metadata(gt_raw[i]))

        # 对每个采样集合计算指标
        res_list: List[Dict[str, float]] = []
        for i in range(sample_count):
            res = self.evaluate_predictions(preds[i], gts, len(samples), metadata, task_name)
            res_list.append(res)

        # 计算 mean 与 std
        res_mean: Dict[str, float] = {}
        res_std: Dict[str, float] = {}
        if res_list:
            for key in res_list[0].keys():
                vals = [float(r[key]) for r in res_list]
                res_mean[key] = statistics.mean(vals)
                res_std[key] = statistics.stdev(vals) if len(vals) > 1 else 0.0

        final_res = {
            "mean": res_mean,
            "std": res_std,
            "valid_rate": 1.0 - float(invalid_num) / (len(samples) * sample_count) if len(samples) > 0 else 0.0,
        }

        return final_res
    
    def record_single_result(self, pred: Optional[str], gt: Optional[Any], metadata: Any, task_name: str) -> Dict[str, Any]:
        if pred is None or gt is None:
            return {}
        return self.evaluate_predictions([pred], [gt], 1, [metadata], task_name)

    def record_results(
        self,
        model_name: str,
        sample_count: int,
        gt_path: str,
        logs_dir: str,
        task_name: str,
    ) -> pd.DataFrame:

        log_dir = os.path.join(logs_dir, task_name)
        if not os.path.exists(log_dir):
            self.logger.warning(f"log not found: {log_dir}")
            return pd.DataFrame()
        
        samples = self._load_samples(log_dir, model_name)
        gt_raw = self._load_gt_raw(gt_path, task_name)

        if len(samples) != len(gt_raw):
            raise ValueError(f"sample count {len(samples)} does not match gt count {len(gt_raw)}")

        # 每个 sample_index 对应一组行（list of dict）
        rows_per_index: List[List[Dict[str, Optional[float]]]] = [[] for _ in range(sample_count)]
        metric_name_order: List[str] = []

        for i, sample in enumerate(samples):
            gt = self.extract_gt(gt_raw[i], task_name)
            metadata = self.prepare_metadata(gt_raw[i])

            # 处理单预测格式
            if 'result' in sample:
                if sample_count > 1:
                    raise ValueError("sample_count should be 1 when result is in sample")
                pred = self.extract_answer(sample['result'], task_name)

                single = self.record_single_result(pred, gt, metadata, task_name) or {}
                if isinstance(sample['result'], str):
                    single['output_length'] = len(sample['result'])
                    single['cot_length'] = self._cot_length(sample['result'])
                
                # 维护 metric 名顺序
                for k in single.keys():
                    if k not in metric_name_order:
                        metric_name_order.append(k)
                rows_per_index[0].append(single)

            # 处理多预测格式
            elif 'results' in sample:
                if sample_count == 1:
                    raise ValueError("sample_count should be greater than 1 when results is in sample")
                results = sample['results']
                if isinstance(results, str):
                    try:
                        results = json.loads(results)
                    except Exception as e:
                        self.logger.debug(f"failed to json.loads results at index {i}: {e}")
                        # 对该 sample 的每个期望预测记录 None
                        for k_idx in range(sample_count):
                            rows_per_index[k_idx].append(single)
                        continue

                # 若结果长度与 sample_count 不一致，则为每个期望位置记录 None
                if not isinstance(results, list) or len(results) != sample_count:
                    self.logger.debug(f"results length mismatch at index {i}: expected {sample_count}, got {len(results) if isinstance(results, list) else type(results)}")
                    for k_idx in range(sample_count):
                        rows_per_index[k_idx].append(single)
                    continue

                # 正常处理每个预测
                for j in range(sample_count):
                    try:
                        pred = self.extract_answer(results[j], task_name)
                    except Exception as e:
                        self.logger.debug(f"extract_answer failed (results[{j}]) for index {i}: {e}")
                        pred = None

                    single = self.record_single_result(pred, gt, metadata, task_name) or {}
                    if isinstance(results[j], str):
                        single['output_length'] = len(results[j])
                        single['cot_length'] = self._cot_length(results[j])
                    
                    for k in single.keys():
                        if k not in metric_name_order:
                            metric_name_order.append(k)
                    rows_per_index[j].append(single)

            else:
                raise ValueError("sample should have either 'result' or 'results'")

        # 将 rows 转为 DataFrame（按 metric_name_order 列顺序）
        dataframes: List[pd.DataFrame] = []
        for idx in range(sample_count):
            rows = rows_per_index[idx]
            if not rows:
                # 空表，保持列信息（如果已有 metric_name_order）
                if metric_name_order:
                    df = pd.DataFrame(columns=metric_name_order)
                else:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame(rows)
                # 保证列顺序，缺失列会被创建为 NaN
                if metric_name_order:
                    # 将现有列 reindex 到 metric_name_order 顺序（新增列保持 NaN）
                    df = df.reindex(columns=metric_name_order)
            dataframes.append(df)

        return pd.concat(dataframes, axis=0, ignore_index=True)

class MolSimiliarityTaskEvaluator(BaseTaskEvaluator):
    def __init__(self):
        super().__init__()
        from core.evaluator import MoleculeSMILESEvaluator
        self.evaluator = MoleculeSMILESEvaluator()
    def extract_gt(self, gt_raw_item: Dict[str, Any], task_name: str) -> Any:
        meta = gt_raw_item['meta']
        meta = json.loads(meta)
        return meta['reference']
    
    def evaluate_predictions(self, preds: List[List[str]], gts: List[Any], total_len: int, metadata: Optional[List[List[Dict[str, Any]]]] = None, task_name = None) -> Dict[str, float]:
        res = self.evaluator.evaluate(preds, gts)
        fts = (res['rdk_sims'] + res['maccs_sims'] + res['morgan_sims']) / 3
        res['fts'] = fts
        return res
    
    def prepare_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return None
    
class TextSimiliarityTaskEvaluator(BaseTaskEvaluator):
    def __init__(self):
        super().__init__()
        from core.evaluator import MoleculeCaptionEvaluator
        self.evaluator = MoleculeCaptionEvaluator()
        
    def extract_gt(self, gt_raw_item, task):
        return gt_raw_item['gt']
    
    def evaluate_predictions(self, preds, gts, total_len, metadata = None, task_name = None):
        res = self.evaluator.evaluate(preds, gts)
        return res

    def prepare_metadata(self, sample):
        return None

class AllCoTTextSimiliarityTaskEvaluator(TextSimiliarityTaskEvaluator):
    def extract_answer(self, pred, task):
        return pred
    
    def extract_gt(self, gt_raw_item, task):
        return gt_raw_item['cot_reference']

class TextExactMatchTaskEvaluator(BaseTaskEvaluator):
    def __init__(self):
        super().__init__()

    def extract_gt(self, gt_raw_item, task):
        return gt_raw_item['gt']

    def evaluate_predictions(self, preds, gts, total_len, metadata = None, task_name = None):
        def match(pred, gt):
            if (not pred) or (not gt):
                return None
            return pred.lower() == gt.lower()
        res = [match(pred, gt) for pred, gt in zip(preds, gts)]
        return {
            "accuracy": sum([1 for r in res if r]) / total_len,
            "valid-rate": sum([1 for r in res if r is not None]) / total_len
        }
    def prepare_metadata(self, sample):
        return None