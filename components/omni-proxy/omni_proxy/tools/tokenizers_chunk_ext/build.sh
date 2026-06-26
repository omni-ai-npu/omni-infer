# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
pip install -U maturin
maturin build --release
pip install target/wheels/tokenizers_chunk_ext-*.whl

#更新代码必须重新安装，重新安装前需要先执行卸载命令或者更新版本号，否则不会重新安装
#pip uninstall -y tokenizers_chunk_ext 