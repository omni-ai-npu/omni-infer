# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
set -e
BUILD_ROOT="$(dirname $(dirname "$(realpath "$0")"))"
cd $BUILD_ROOT/
pip install -e . "$@"