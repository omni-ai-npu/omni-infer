# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Source CANN environment
# source /usr/local/Ascend/ascend-toolkit/set_env.sh

# compile to a package
g++ -fPIC -shared -std=c++11 \
    -I${ASCEND_TOOLKIT_HOME}/include \
    -L${ASCEND_TOOLKIT_HOME}/lib64 -lascendcl \
    tensor_register.cpp -o tensor_register.so

echo "Build completed: tensor_register.so"

python setup.py build_ext --inplace -j8
