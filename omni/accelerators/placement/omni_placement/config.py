# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import yaml
from pathlib import Path


class Config:
    def __init__(self, config_yaml_path):
        omni_config = self.load_and_validate_config(config_yaml_path)
        if omni_config:
            self._convert_dict_to_obj(omni_config)

    @staticmethod
    def load_and_validate_config(config_yaml_path):
        try:
            # 获取文件的绝对路径
            config_yaml_path = Path(config_yaml_path).absolute()
            print(f"Attempting to read YAML file: {config_yaml_path}")
            # 打开文件并读取内容
            with open(config_yaml_path, mode='r', encoding='utf-8') as fh:
                omni_config = yaml.safe_load(fh)
                print(f"Successfully loaded YAML file content:\n{yaml.dump(omni_config, allow_unicode=True, sort_keys=False)}")
            return omni_config
        except FileNotFoundError:
            print(f"File not found: {config_yaml_path}")
        except yaml.YAMLError as e:
            print(f"YAML parsing error: {e}")
            try:
                with open(config_yaml_path, mode='r', encoding='utf-8') as fh:
                    raw_content = fh.read()
                    print(f"Raw YAML file content:\n{raw_content}")
            except Exception as inner_e:
                print(f"Unable to read raw YAML content: {inner_e}")
        return None

    def _convert_dict_to_obj(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                # 如果值是字典，递归转换为具有属性的对象
                sub_obj = Config.__new__(Config)
                sub_obj._convert_dict_to_obj(value)
                setattr(self, key, sub_obj)
            else:
                setattr(self, key, value)
    
    def getattr(self, key, default_value):
        return getattr(self, key, default_value)
