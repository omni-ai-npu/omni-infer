#!/usr/bin/env python3
# 标准库导入
import argparse
import ast
import base64
import filecmp
import gzip
import json
import os
import pickle
import random
import re
import shutil
import string
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path

# 第三方库导入
import numpy as np
import requests


FILE_EXTEN_LEN = 4
DIM_THREE = 3
INCLUDE_USAGE = False
IGNORE_EOS = True


# ==================== API客户端部分 ====================

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=length))
    return random_string


class APIClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def make_request(self, base_url, endpoint, data, stream=False):
        url = f"{base_url}{endpoint}"
        data["stream"] = stream
        if IGNORE_EOS:
            data["ignore_eos"] = True
        if stream and INCLUDE_USAGE:
            data["stream_options"] = {
                "include_usage": True, 
                "continuous_usage_stats": True
            }
        try:
            if stream:
                return self._handle_stream_request(url, data)
            else:
                return self._handle_normal_request(url, data)
        except requests.exceptions.RequestException as e:
            return {"error": f"请求失败: {str(e)}", "request_id": None}

    def _handle_normal_request(self, url, data):
        response = self.session.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        result["request_id"] = result.get("id")
        return result

    def _handle_stream_request(self, url, data):
        response = self.session.post(url, json=data, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            decoded_line = line.decode('utf-8')
            if not decoded_line.startswith('data:'):
                continue
            data_str = decoded_line[len('data:'):].strip()
            if data_str == '[DONE]':
                yield {"data": "[DONE]", "request_id": None}
                break
            try:
                chunk = json.loads(data_str)
                chunk["request_id"] = chunk.get("id")
                yield chunk
            except json.JSONDecodeError:
                continue

    def close(self):
        self.session.close()


class UnifiedAPIHandler:
    ENDPOINT_MAP = {
        "completion_stream": "/v1/completions",
        "completion_normal": "/v1/completions",
        "chat_completion_stream": "/v1/chat/completions",
        "chat_completion_normal": "/v1/chat/completions"
    }

    def __init__(self):
        self.client = APIClient()

    def _format_messages(self, messages):
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return messages

    def call_api(self, request_type, ip="127.0.0.1", port=8001, model="Pangu72B",
                 prompt=None, messages=None, max_tokens=20, **kwargs):
        if request_type not in self.ENDPOINT_MAP:
            raise ValueError(f"不支持的请求类型: {request_type}")
        
        data = {"model": model, "max_tokens": max_tokens, **kwargs}
        data["stream"] = request_type.endswith("_stream")
        if prompt is not None:
            data["prompt"] = prompt
        if messages is not None:
            data["messages"] = self._format_messages(messages)
        
        return self.client.make_request(
            f"http://{ip}:{port}", 
            self.ENDPOINT_MAP[request_type], 
            data, 
            data["stream"]
        )

    def _make_api_call(self, request_type, ip, port, content_key, content, **kwargs):
        kwargs[content_key] = content
        return self.call_api(request_type, ip, port, **kwargs)

    def completion_stream(self, ip, port, prompt, **kwargs):
        return self._make_api_call("completion_stream", ip, port, "prompt", prompt, **kwargs)

    def completion_normal(self, ip, port, prompt, **kwargs):
        return self._make_api_call("completion_normal", ip, port, "prompt", prompt, **kwargs)

    def chat_completion_stream(self, ip, port, messages, **kwargs):
        return self._make_api_call("chat_completion_stream", ip, port, "messages", messages, **kwargs)

    def chat_completion_normal(self, ip, port, messages, **kwargs):
        return self._make_api_call("chat_completion_normal", ip, port, "messages", messages, **kwargs)

    def close(self):
        self.client.close()


class APIProcessor:
    def __init__(self):
        self.handler = UnifiedAPIHandler()

    def _get_routed_experts_from_chunk(self, chunk):
        if "choices" in chunk and chunk["choices"]:
            choice = chunk["choices"][0]
            if "routed_experts" in choice:
                return choice["routed_experts"]
        return None

    def _process_stream_response(self, response_generator):
        stream_arrays = []
        request_id = None
        all_chunks = []
        for chunk in response_generator:
            if chunk.get("data") == "[DONE]":
                break
            all_chunks.append(chunk)
            routed_experts_data = self._get_routed_experts_from_chunk(chunk)
            if routed_experts_data:
                routed_experts = self._decode_numpy_array(
                    routed_experts_data["routed_experts_str"],
                    routed_experts_data["routed_experts_shape"],
                    routed_experts_data["routed_experts_dtype"]
                )
                stream_arrays.append(routed_experts)
            if not request_id and "request_id" in chunk:
                request_id = chunk["request_id"]
        concatenated_array = None
        if stream_arrays:
            concatenated_array = list(chain.from_iterable(stream_arrays))
        return concatenated_array, all_chunks

    def _process_normal_response(self, response):
        routed_experts = self._get_routed_experts_from_chunk(response)
        routed_experts = routed_experts if routed_experts else {}
        arr = self._decode_numpy_array(
            routed_experts.get("routed_experts_str"),
            routed_experts.get("routed_experts_shape"),
            routed_experts.get("routed_experts_dtype")
        )
        return arr, response

    @staticmethod
    def _decode_numpy_array(encoded_str, shape, dtype):
        if encoded_str:
            decoded_bytes = base64.b64decode(encoded_str)
            decompressed_bytes = zlib.decompress(decoded_bytes)
            arr = np.frombuffer(decompressed_bytes, dtype=dtype)
            return arr.reshape(shape).tolist()
        else:
            print("请求内容中没有routed_experts")
            return None

    @staticmethod
    def _save_3d_array_to_txt(array, filename):
        if not isinstance(array, list) or not array:
            raise ValueError("必须是三维数组")
        if filename.startswith("cmpl"):
            pos = len(filename) - FILE_EXTEN_LEN
            filename = filename[:pos] + "-0" + filename[pos:]
        os.makedirs("saved_req_arrays", exist_ok=True)
        filepath = os.path.join("saved_req_arrays", filename)
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(str(array).replace(' ', ''))
        return filepath

    def process_and_save(self, request_type, ip, port, content, model="Pangu72B",
                        max_tokens=20, save_to_file=True, return_full_response=False, **kwargs):
        result = {
            "success": False, 
            "request_type": request_type, 
            "request_id": None,
            "concatenated_array": None, 
            "saved_file": None, 
            "error": None
        }
        
        try:
            method = getattr(self.handler, request_type)
            response = method(ip, port, content, model=model, max_tokens=max_tokens, **kwargs)
            
            is_stream = request_type.endswith("_stream")
            if is_stream:
                concatenated_array, all_chunks = self._process_stream_response(response)
                result["concatenated_array"] = concatenated_array
                if all_chunks:
                    result["request_id"] = all_chunks[0].get("request_id")
                if return_full_response:
                    result["full_response"] = all_chunks
            else:
                concatenated_array, full_response = self._process_normal_response(response)
                result["concatenated_array"] = concatenated_array
                result["request_id"] = full_response.get("request_id")
                if return_full_response:
                    result["full_response"] = full_response

            if save_to_file and result["concatenated_array"] and result["request_id"]:
                try:
                    filepath = self._save_3d_array_to_txt(
                        result["concatenated_array"], 
                        f"{result['request_id']}.txt"
                    )
                    result["saved_file"] = filepath
                except Exception as e:
                    result["error"] = f"保存文件时出错: {str(e)}"
                    print(f"处理数组时出错: {e}")
            
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
            print(f"处理请求时出错: {e}")
        return result

    def get_and_save_array(self, request_type, ip, port, content, model="Pangu72B", 
                          max_tokens=20, **kwargs):
        result = self.process_and_save(
            request_type, ip, port, content, model, max_tokens,
            save_to_file=True, return_full_response=False, **kwargs
        )
        if result["success"] and result["saved_file"]:
            return result["saved_file"]
        else:
            error_msg = result.get("error", "未知错误")
            if error_msg is not None:
                raise ValueError(f"请求失败或保存失败: {error_msg}")
            else:
                raise ValueError(f"未保存请求数据")

    def close(self):
        self.handler.close()


def run_all_request(ip_addr, port, max_tokens, model_name):
    """执行all_request.py的功能"""
    processor = APIProcessor()
    try:
        print("=== 流式Completion请求 ===")
        result1 = processor.process_and_save(
            "completion_stream", ip_addr, port, "在很久很久以前", model_name, max_tokens
        )
        if result1["success"]:
            print(f"请求ID: {result1['request_id']}")
            if result1["saved_file"]:
                print(f"文件已保存: {result1['saved_file']}")
        time.sleep(0.01)
        
        print("\n=== 非流式Chat Completion请求（字符串格式） ===")
        result2 = processor.process_and_save(
            "chat_completion_normal", ip_addr, port, "详细介绍一下上海", model_name, max_tokens
        )
        if result2["success"]:
            print(f"请求ID: {result2['request_id']}")
        time.sleep(0.01)
        
        print("\n=== completion_normal 调用（非流式文本补全） ===")
        result3 = processor.get_and_save_array(
            "completion_normal", ip_addr, port, "人工智能是", model_name, max_tokens
        )
        print(f"文件已保存: {result3}")
        time.sleep(0.01)
        
        print("\n=== chat_completion_stream 调用（流式文本补全） ===")
        result4 = processor.get_and_save_array(
            "chat_completion_stream", ip_addr, port, "请写一部小说", model_name, max_tokens
        )
        print(f"文件已保存: {result4}")
    finally:
        processor.close()


# ==================== 数据处理部分 ====================

def merge(dict1, dict2):
    result = {}
    for key in dict1.keys() | dict2.keys():
        val1 = dict1.get(key, {})
        val2 = dict2.get(key, {})
        for key_sub in val1.keys() | val2.keys():
            value = val2.get(key_sub, val1.get(key_sub, None))
            if key not in result:
                result[key] = {}
            result[key][key_sub] = value
    return result


def read_pkl(key_word="data_"):
    dir_path = './test'
    data_files = [f for f in os.listdir(dir_path) if key_word in f and f.endswith('.pkl')]
    data = {}
    for filename in data_files:
        filepath = os.path.join(dir_path, filename)
        try:
            with gzip.open(filepath, 'rb') as f:
                file_data = pickle.load(f)
                data = merge(data, file_data)
        except Exception as e:
            print(f"读取文件 {filepath} 时出错: {e}")
    return data


def save_data():
    list_data = read_pkl("indices_")
    dict_data = read_pkl("data_")
    os.makedirs("dump_arrays", exist_ok=True)
    for request_id, indices_dict in list_data.items():
        for part_key, indices in indices_dict.items():
            data = []
            for i in indices:
                if i in dict_data[part_key]:
                    data.append(dict_data[part_key][i].tolist())
                else:
                    print(f"Key {i} not found in dict_data.")
            prefix = part_key + '-' if part_key != 'total' else ''
            filename = os.path.join("dump_arrays", prefix + f'{request_id}.txt')
            with open(filename, 'w') as f:
                f.write(str(data).replace(' ', ''))


# ==================== 文件合并部分 ====================

def find_prefix_files(directory, prefixes):
    try:
        files = os.listdir(directory)
    except (FileNotFoundError, PermissionError) as e:
        print(f"访问目录失败: {e}")
        return {}
    
    file_groups = {}
    for f in files:
        if not f.endswith('.txt'):
            continue
        for p in prefixes:
            if f.startswith(p):
                base_name = f[len(p):-FILE_EXTEN_LEN]
                file_groups.setdefault(base_name, {})[p] = os.path.join(directory, f)
                break
    
    return {
        base_name: (prefix_dict['prefill-'], prefix_dict['decode-']) 
        for base_name, prefix_dict in file_groups.items() 
        if 'prefill-' in prefix_dict and 'decode-' in prefix_dict
    }


def parse_array_content(content):
    try:
        content = re.sub(r',\s*]', ']', re.sub(r',\s*,', ',', content.strip()))
        return np.array(ast.literal_eval(content), dtype=int)
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"解析数组内容失败: {e}")
        raise


def merge_3d_arrays(arr1, arr2):
    if arr1.ndim != DIM_THREE or arr2.ndim != DIM_THREE:
        raise ValueError("输入数组必须是三维数组")
    if arr1.shape[1:] != arr2.shape[1:]:
        raise ValueError("数组的第二维和第三维必须相同")
    return np.concatenate((arr1, arr2), axis=0)


def save_merged_array(array, filename):
    try:
        array_str = np.array2string(array, separator=',', threshold=np.inf, max_line_width=np.inf)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(re.sub(r'\s+', '', array_str))
    except IOError as e:
        print(f"保存文件失败: {e}")
        raise


def process_files(directory="."):
    file_pairs = find_prefix_files(directory, ['prefill-', 'decode-'])
    if not file_pairs:
        return
    
    success = 0
    for base_name, (prefill_path, decode_path) in file_pairs.items():
        try:
            with open(prefill_path, 'r', encoding='utf-8') as f1, \
                 open(decode_path, 'r', encoding='utf-8') as f2:
                prefill_array = parse_array_content(f1.read())
                decode_array = parse_array_content(f2.read())
            
            merged = merge_3d_arrays(prefill_array, decode_array)
            output_file = os.path.join(directory, f"{base_name}.txt")
            save_merged_array(merged, output_file)
            success += 1
        except Exception as e:
            print(f"处理 {base_name} 失败: {e}")
    
    print(f"合并处理完成，成功: {success}/{len(file_pairs)}")


# ==================== 主测试部分 ====================

def parse_args():
    parser = argparse.ArgumentParser(description="发送请求以验证导出专家id功能的正确性")
    parser.add_argument("-n", "--num-processes", type=int, default=3, 
                       help="并行执行的进程数")
    parser.add_argument("--save-tmp-dir", action="store_true", 
                       help="保留中间临时文件")
    parser.add_argument("--read-only", action="store_true", 
                       help="仅运行读取dump数据的功能")
    parser.add_argument("--ip-addr", type=str, default="127.0.0.1", 
                       help="API服务器IP地址")
    parser.add_argument("--port", type=int, default=8001, 
                       help="API服务器端口")
    parser.add_argument("--model-name", type=str, default="Pangu72B", 
                       help="模型名称")
    parser.add_argument("--max-tokens", type=int, default=200, 
                       help="生成的最大token数")
    return parser.parse_args()


def setup_paths(mode="create"):
    export_path = os.environ.get("EXPORT_MOE_EXPERTS_TEST_PATH", None)
    if export_path is None:
        raise ValueError("环境变量 EXPORT_MOE_EXPERTS_TEST_PATH 未设置")
    export_path = Path(export_path)
    current_dir = Path.cwd()
    
    paths = [
        export_path, 
        current_dir / "dump_arrays", 
        current_dir / "saved_req_arrays", 
        current_dir / "test"
    ]
    
    if mode == "create":
        for p in paths:
            p.mkdir(parents=True, exist_ok=True)
    elif mode == "delete":
        for p in paths:
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
    else:
        raise ValueError("mode 必须为 'create' 或 'delete'")
    
    return export_path, current_dir


def cleanup(export_path, current_dir):
    patterns = [
        (export_path, "*.pkl"), 
        (current_dir / "dump_arrays", "*"),
        (current_dir / "saved_req_arrays", "*"), 
        (current_dir, "*.txt"),
        (current_dir / "test", "*")
    ]
    for base, pattern in patterns:
        for f in base.glob(pattern):
            try:
                if f.is_file():
                    f.unlink()
                else:
                    shutil.rmtree(f)
            except Exception:
                pass


def run_scripts_parallel(max_workers=3, ip_addr="127.0.0.1", port=8001, max_tokens=20, model_name = "Pangu72B"):
    """并行执行脚本"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_all_request, ip_addr, port, max_tokens, model_name) 
            for _ in range(max_workers)
        ]

        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"脚本执行失败: {str(e)[:100]}")


def process_results(export_path, current_dir):
    test_dir = current_dir / "test"
    for f in export_path.iterdir():
        if f.is_file():
            shutil.copy2(f, test_dir)
    save_data()
    process_files(str(current_dir / "dump_arrays"))


def compare_dirs(saved_dir, dump_dir):
    has_diff = 0
    for saved_file in saved_dir.rglob("*.txt"):
        dump_file = dump_dir / saved_file.relative_to(saved_dir)
        if not dump_file.exists():
            print(f"文件不存在: {saved_file.relative_to(saved_dir)}")
            has_diff += 1
        elif not filecmp.cmp(saved_file, dump_file, shallow=False):
            print(f"内容不同: {saved_file.relative_to(saved_dir)}")
            has_diff += 1
    return has_diff


def main():
    try:
        args = parse_args()
        export_path, current_dir = setup_paths()
        
        print("开始清理和运行Python脚本...")
        if not args.read_only:
            cleanup(export_path, current_dir)
            run_scripts_parallel(args.num_processes, args.ip_addr, args.port, args.max_tokens, args.model_name)
        process_results(export_path, current_dir)
        has_diff = 0
        print("Python脚本执行完成")

        if not args.read_only:
            print(f"比较 {current_dir/'saved_req_arrays'} 和 {current_dir/'dump_arrays'} 目录下的txt文件...")
            has_diff = compare_dirs(current_dir / "saved_req_arrays", current_dir / "dump_arrays")
            print("比较完成!")
            print("通过测试" if has_diff == 0 else "测试失败")

            if not args.save_tmp_dir:
                setup_paths("delete")
        sys.exit(1 if has_diff else 0)
    except subprocess.CalledProcessError as e:
        print(f"子进程执行失败: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n程序中断")
        sys.exit(130)
    except Exception as e:
        print(f"程序错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()