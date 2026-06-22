# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
START_DIR=$PWD

SOURCE_ROOT="$(dirname $(dirname "$(realpath "$0")"))"

# test omni-placemnt
cd $SOURCE_ROOT/omni/accelerators/placement
pytest tests/test_moe_weights.py::TestGetExpertIds