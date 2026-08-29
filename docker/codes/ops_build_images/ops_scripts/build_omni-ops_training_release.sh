#!/bin/bash
set -ex
script_dir=$(dirname "$(readlink -f "$0")")
DIR_PATH=$1
COMPILE_UNIT=$2
project_version=$3
cann_version=$4
ops_local_path_suffix=$5
pta_version=$6

echo "cann_version: ${cann_version}"
lower_cann_version=$(echo "$cann_version" | tr '[:upper:]' '[:lower:]')

time=$(date -u +"%Y%m%d" --date='+8 hours')

echo "COMPILE_UNIT: ${COMPILE_UNIT}"
# source $script_dir/set_env.sh

if [[ "${project_version}" =~ ^release- ]]; then
    type="release"
    # 提取版本号（去除"release-"前缀）
    version="${project_version#release-}"
elif [[ "${project_version}" =~ ^poc- ]]; then
    type="poc"
    # 提取版本号（去除"poc-"前缀）
    version="${project_version#poc-}"
else
    echo "未知类型: ${project_version}"
    return 1
fi
echo "类型: $type, 版本: $version"

if [ ${COMPILE_UNIT} == "ascend910b" ]; then
  bias_name="910B"
elif [ ${COMPILE_UNIT} == "ascend910_93" ]; then
  bias_name="910C"
elif [ ${COMPILE_UNIT} == "ascend950" ]; then
  bias_name="910D"
fi

# 修改版本名称
if [ ${type} == "release" ]; then
  sed -i "s#CANN-omni_training_custom_ops-\${CANN_VERSION}-linux.\${CMAKE_SYSTEM_PROCESSOR}#cann${bias_name}-omni_training_custom_ops-${version}-${cann_version}-linux-\${CMAKE_SYSTEM_PROCESSOR}#g" $DIR_PATH/training/ascendc/CMakeLists.txt

  sed -i "s#name=\"omni_training_custom_ops\"#name=\"omni_training_ascendc_custom_ops\"#g" $DIR_PATH/training/ascendc/torch_ops_extension/setup.py
  sed -i "s#version='1.0'#version='${version}+${lower_cann_version}.pta${pta_version}'#g" $DIR_PATH/training/ascendc/torch_ops_extension/setup.py
  sed -i "s#1.0.0#${version}#g" $DIR_PATH/pyproject_training_triton.toml || true
  triton_tar_name="omni_training_triton_custom_ops_${version}"
  pypto_tar_name="omni_training_pypto_custom_ops_${version}"
else
  sed -i "s#CANN-omni_training_custom_ops-\${CANN_VERSION}-linux.\${CMAKE_SYSTEM_PROCESSOR}#cann${bias_name}-omni_training_custom_ops-${version}-${cann_version}-linux-\${CMAKE_SYSTEM_PROCESSOR}-${time}#g" $DIR_PATH/training/ascendc/CMakeLists.txt

  sed -i "s#name=\"omni_training_custom_ops\"#name=\"omni_training_ascendc_custom_ops\"#g" $DIR_PATH/training/ascendc/torch_ops_extension/setup.py
  sed -i "s#version='1.0'#version='${version}.dev${time}+${lower_cann_version}.pta${pta_version}'#g" $DIR_PATH/training/ascendc/torch_ops_extension/setup.py
  sed -i "s#1.0.0#${version}.dev${time}#g" $DIR_PATH/pyproject_training_triton.toml  || true
  triton_tar_name="omni_training_triton_custom_ops_${version}_${time}"
  pypto_tar_name="omni_training_pypto_custom_ops_${version}_${time}"
fi

# triton源码打包
cd $DIR_PATH/training
tar -cvf ${triton_tar_name}.tar triton/ || true
zip -r ${triton_tar_name}.zip triton/ || true
tar -czvf ${triton_tar_name}.tar.gz triton/ || true

# pypto源码打包
cd $DIR_PATH/training
tar -cvf ${pypto_tar_name}.tar pypto/ || true
zip -r ${pypto_tar_name}.zip pypto/ || true
tar -czvf ${pypto_tar_name}.tar.gz pypto/ || true

export http_proxy=http://p_llmdevops:4A_dq%21R_@proxy.huawei.com:8080/
export https_proxy=http://p_llmdevops:4A_dq%21R_@proxy.huawei.com:8080/

# 自定义算子编译
echo "编译自定义算子"
ops=$(python3 ${script_dir}/scan_op.py ${DIR_PATH} training ${COMPILE_UNIT})

cd $DIR_PATH/training/ascendc
#bash build.sh --compute-unit ${COMPILE_UNIT}
#bash build.sh -n "${ops}" --compute-unit "${COMPILE_UNIT}"

for i in {1..3}; do
    bash build.sh --compute-unit "${COMPILE_UNIT}" && break

    echo "build.sh execute failed, retry ${i}/3"

    sleep 30

    if [ "$i" -eq 3 ]; then
        echo "build.sh 执行失败！"
        exit 1
    fi
done

# 自定义算子安装
cd $DIR_PATH/training/ascendc/output
# chmod +x CANN-omni_custom_ops-*.run
chmod +x cann${bias_name}-*.run
#./CANN-omni_custom_ops-*.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp
#source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash

# torch包编译安装
cd $DIR_PATH/training/ascendc/torch_ops_extension
bash build_and_install.sh

# triton包编译
cd $DIR_PATH
bash build_triton_wheels.sh || true


if ! ls $DIR_PATH/training/ascendc/output/cann${bias_name}-*.run >/dev/null 2>&1 ; then
  echo "omni-ops编译失败！"
  exit 1
fi

if ! ls $DIR_PATH/training/ascendc/torch_ops_extension/dist/omni_training_ascendc_custom_ops-*.whl >/dev/null 2>&1 ; then
  echo "omni-ops编译失败！"
  exit 1
fi

package_path="$DIR_PATH/../../ops-packages/${ops_local_path_suffix}/"
mkdir -p $package_path
# cp $DIR_PATH/training/ascendc/output/CANN-omni_custom_ops-*.run $package_path || true
# cp $DIR_PATH/training/ascendc/torch_ops_extension/dist/omni_custom_ops-*.whl $package_path || true
cp $DIR_PATH/training/ascendc/output/cann${bias_name}-*.run $package_path || true
cp $DIR_PATH/training/ascendc/torch_ops_extension/dist/omni_training_ascendc_custom_ops-*.whl $package_path || true
cp $DIR_PATH/dist/omni_training_triton_custom_ops-*.whl $package_path || true
cp $DIR_PATH/training/${triton_tar_name}.* $package_path || true
cp $DIR_PATH/dist/omni_training_pypto_custom_ops-*.whl $package_path || true
cp $DIR_PATH/training/${pypto_tar_name}.* $package_path || true