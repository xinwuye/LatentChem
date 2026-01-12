import json
import regex as re
from typing import Literal, Optional

def tranform_str_to_json(str_input):
    if str_input is None:
        return None
    ## 假如LLM输出的是类似json的字符串, 我需要设定一个逻辑, 把字符串重新转换成json
    ## o1-mini的感觉, 是要移除字符串里面的\n，并且把所有的\"都改成 "
    if "</think>\n\n" in str_input:
        str_input = str_input.split("</think>\n\n")[-1]
        
    if "```json\n" in str_input:
        str_input = str_input.split("```json\n")[1]
        str_input = str_input.replace("\n```", '')
    
    unescaped_str = str_input.replace('\n    ', '').replace('\n', '').replace('\"', '"')
    
    pattern = re.compile(
        r'("Major Product"\s*:\s*"(?:\\.|[^"\\])*"\s*)(?=\s*"Byproduct\(s\)")',
        flags=re.DOTALL
    )
    
    '''{
    "Major Product": "CC(C)(C)OC(=O)N[C@H](C=O)CC1CCC1""Byproduct(s)": "CCOC(C)=O.CC(=O)O"
    }'''

    unescaped_str = re.sub(pattern, r'\1,', unescaped_str)
    
    try:
        json_obj = json.loads(unescaped_str)
    except json.JSONDecodeError as e:
        return None
    
    def _replace_outputs(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                if k == 'outputs':
                    if isinstance(v, list):
                        new_value = v[0] if len(v) > 0 else None
                    else:
                        new_value = v
                    new_obj[k] = new_value
                else:
                    new_obj[k] = v
            return new_obj
        else: return obj

    processed = _replace_outputs(json_obj)
    return processed

answer_pattern = re.compile(r'<answer\s*>(.*?)</answer\s*>', flags=re.S)

def extract_answer(str_input):
    matches = re.findall(answer_pattern, str_input)
    last_content = matches[-1].strip() if matches else None
    return last_content

def parse_raw_response(
    raw_response: str,
    field: str,
    format: Literal["str", "int", "float", "bool"] = "str"
) -> Optional[str]:
    """
    从 JSON 格式字符串中提取指定字段的值，忽略开头的 <think>...</think> 部分，
    并根据 format 参数验证值的类型。

    Args:
        raw_response (str): 包含 JSON 数据的字符串，可能以 <think>...</think> 开头。
        field (str): 要提取的字段名（如 "count"）。
        format (Literal["str", "int", "float", "bool"]): 期望的返回值类型，默认为 "str"。

    Returns:
        Optional[str]: 字段的值（字符串形式），如果未找到或类型不匹配则返回 None。
    """
    # 1. 移除 <think>...</think> 部分（如果有）
    cleaned_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)

    # 2. 尝试匹配带引号的字符串值（如 "count": "2"）
    quoted_pattern = rf'"{field}":\s*"([^"]+)"'
    match = re.search(quoted_pattern, cleaned_response)
    if match:
        value = match.group(1)
        if format == "str":
            return value
        return _validate_format(value, format)

    # 3. 尝试匹配不带引号的值（如 "count": 2, "active": true）
    unquoted_pattern = rf'"{field}":\s*([^,}}\s]+)'
    match = re.search(unquoted_pattern, cleaned_response)
    if match:
        value = match.group(1).strip()
        return _validate_format(value, format)

    # 4. 未找到字段
    return None

def _validate_format(value: str, format: str) -> Optional[str]:
    """
    验证值的类型是否符合指定的 format。

    Args:
        value (str): 提取的原始值（字符串形式）。
        format (str): 期望的类型（"str"、"int"、"float"、"bool"）。

    Returns:
        Optional[str]: 转换后的值（字符串形式），如果类型不匹配则返回 None。
    """
    try:
        if format == "int":
            int(value)
            return value
        elif format == "float":
            float(value)
            return value
        elif format == "bool":
            if value.lower() in ("true", "false"):
                return value.lower()
            return None
        elif format == "str":
            return value
        return None
    except (ValueError, TypeError):
        return None

def _combine_list(raw):
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                raw = '.'.join(parsed)
            # 如果解析成功但不是 list，什么都不做
        except (json.JSONDecodeError, TypeError):
            # 不能解析为 JSON，什么都不做
            pass
    elif isinstance(raw, list):
        raw = '.'.join(raw)
    return raw