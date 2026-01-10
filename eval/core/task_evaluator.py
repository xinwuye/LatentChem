import os
import json
import logging
import statistics
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

from core.utils import extract_answer

logger = logging.getLogger(__name__)


class BaseTaskEvaluator(ABC):
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def extract_gt(self, gt_raw_item: Any) -> Any:
        """从单个 gt_raw[i] 提取 gt（可覆盖）。"""
        raise NotImplementedError

    @abstractmethod
    def evaluate_predictions(self, preds: List[List[str]], gts: List[Any], total_len: int, metadata: Optional[List[List[Dict[str, Any]]]] = None) -> Dict[str, float]:
        """对给定 preds/gts 计算评价指标（可覆盖）。
        返回字典形式的指标，例如 {"bleu": 0.8, ...}
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
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

    def run(
        self,
        model_name: str,
        sample_count: int,
        gt_path: str,
        logs_dir: str,
        results_dir: str,
        task_name: str,
    ) -> Dict[str, Any]:
        log_dir = os.path.join(logs_dir, task_name)
        if not os.path.exists(log_dir):
            raise ValueError(f"logs_dir {log_dir} is not correct")

        samples = self._load_samples(log_dir, model_name)
        gt_raw = self._load_gt_raw(gt_path, task_name)
        
        if len(samples) != len(gt_raw):
            raise ValueError(f"sample count {len(samples)} does not match gt count {len(gt_raw)}")

        preds: List[List[str]] = [[] for _ in range(sample_count)]
        metadata: List[List[Dict[str, Any]]] = [[] for _ in range(sample_count)]
        gts: List[Any] = []
        
        invalid_num = 0
        
        for i, sample in enumerate(samples):

            gt = self.extract_gt(gt_raw[i])

            # 支持 sample['result'] 或 sample['results'] 两种格式
            if 'result' in sample:
                if sample_count > 1:
                    raise ValueError("sample_count should be 1 when result is in sample")
                pred = extract_answer(sample['result'])
                if pred is None:
                    invalid_num += 1
                    continue
                preds[0].append(pred)
                meta = self.prepare_metadata(sample)
                metadata[0].append(meta)

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
                    pred = extract_answer(results[j])
                    if pred is None:
                        invalid_num += 1
                        continue
                    preds[j].append(pred)
                    meta = self.prepare_metadata(sample)
                    metadata[j].append(meta)

            else:
                raise ValueError("sample should have either 'result' or 'results'")
                
            gts.append(gt)

        # 对每个采样集合计算指标
        res_list: List[Dict[str, float]] = []
        for i in range(sample_count):
            res = self.evaluate_predictions(preds[i], gts, len(samples), metadata[i])
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
            "raw": res_list,
            "valid_rate": 1.0 - float(invalid_num) / len(samples) if len(samples) > 0 else 0.0,
        }

        # 保存结果
        out_dir = os.path.join(results_dir, task_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"eval_score_{model_name}.json")
        with open(out_path, 'w') as f:
            json.dump(final_res, f, indent=4)

        return final_res

class MolSimiliarityTaskEvaluator(BaseTaskEvaluator):
    def __init__(self):
        from core.evaluator import MoleculeSMILESEvaluator
        self.evaluator = MoleculeSMILESEvaluator()
    def extract_gt(self, gt_raw_item: Dict[str, Any]) -> Any:
        meta = gt_raw_item['meta']
        meta = json.loads(meta)
        return meta['reference']
    
    @abstractmethod
    def evaluate_predictions(self, preds: List[List[str]], gts: List[Any], total_len: int, metadata: Optional[List[List[Dict[str, Any]]]] = None) -> Dict[str, float]:
        res = self.evaluator.evaluate(preds, gts)
        fts = (res['rdk_sims'] + res['maccs_sims'] + res['morgan_sims']) / 3
        res['fts'] = fts
        return res
    
    @abstractmethod
    def prepare_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return None
    
class TextSimiliarityTaskEvaluator(BaseTaskEvaluator):
    def __init__(self):
        from core.evaluator import MoleculeCaptionEvaluator
        self.evaluator = MoleculeCaptionEvaluator()
        
    def extract_gt(self, gt_raw_item):
        return gt_raw_item['gt']
    
    def evaluate_predictions(self, preds, gts, total_len, metadata = None):
        res = self.evaluator.evaluate(preds, gts)
        return res

    def prepare_metadata(self, sample):
        return None
    
class ExactMatchTaskEvaluator(BaseTaskEvaluator):
    def __init__(self):
        pass

    def extract_gt(self, gt_raw_item):
        return gt_raw_item['gt']

    def evaluate_predictions(self, preds, gts, total_len, metadata = None):
        res = self.evaluator.evaluate(preds, gts)
        return res

    def prepare_metadata(self, sample):
        return None