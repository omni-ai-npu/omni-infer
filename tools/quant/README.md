# 简介
本[Optiquant](#optiquant)工具支持deepseek/kimi k2/pangu 718B/qwen 235B/gpt-oss的量化，其他Omni-infer中使用的量化权重由[Modelslim](#modelslim)生成。

## Optiquant
### 编译步骤  
进入python目录下执行： python setup.py bdist_wheel 
进入dist目录下执行: pip install *.whl --force-reinstall

### 操作步骤  
**注意：假如权重为FP8，需要先转为BF16**：  
1、cd ./omniinfer/tools/quant/  
python fp8_cast_bf16.py --input-fp8-hf-path {fp8权重路径} --output-bf16-hf-path {bf16权重路径}  
2、将原始FP8权重目录下***除.safetonsors外的所有文件***覆盖拷贝到BF16权重路径下  
3、新建{量化权重路径}，将原始FP8权重目录下***除.safetonsors外的所有文件***拷贝到{量化权重路径}下（尤其是保证model.safetensors.index.json文件一定要用***FP8权重目录下的***）

4、执行量化命令  

***deepseek/kimi k2/pangu 718B:***

int8量化: python quant_deepseek_kimi2.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu"  
int4量化: python quant_deepseek_kimi2.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "npu" --w4  

***deepseek 3.2:***

int8量化: python quant_deepseek32.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu"

***qwen 235B:***  

int8量化: python quant_qwen.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu"  

***gpt-oss:*** 

gpt-oss-120b int8量化: python quant_gptoss.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu" --model-type "120b"  
gpt-oss-20b int8量化: python quant_gptoss.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu" --model-type "20b"  
mxfp4 to int8量化： bash gpt_oss_mxfp4_to_int8.sh --input-path {mxfp4权重路径} --output-path {量化权重路径}  

***kimi k2 w4 pergroup to perchannel:***

python quant_deepseek_kimi2.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "npu" --pergroup-to-perchannel --w4

***C8量化:***  先拉起服务化dump数据再使用量化工具 

（1）拉起服务化时在config文件中加入c8_calib_path,对话后将knope保存至自定义的c8_calib_path

（2）python quant_deepseek_kimi2.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu" --w4 --c8-calib-path "your_path" --kvs-safetensor-name "your_name"  
    若只想执行c8量化，可以将if args.w4后的部分注释后执行上述命令  

### 参数说明
--input-bf16-hf-path   原始bf16权重路径  
--output-path          生成量化权重路径  
--device               设备类型，支持cpu和npu  
--model-name           hugginface权重名称，在没有元数据配置时自动根据权重名下载配置文件  
--w4                   int4量化标识, 不加该参数时为int8量化  
--pangu-mode           pangu量化标识, 开启时量化pangu 718B权重

## Modelslim
### Modelslim支持的权重
[Modelslim](https://gitcode.com/Ascend/msmodelslim)是一款开源昇腾模型压缩工具，Optiquant的部分功能已经合入该工具。以下模型可以一键生成与Omni-infer推理框架兼容的权重：

**DeepSeek**:
[DeepSeek-V3.1-W4A8C16](./modelslim/DeepSeek-V3.1-W4A8C16.md)
[DeepSeek-V3.2-W4A8C16](./modelslim/DeepSeek-V3.2-W4A8C16.md)

**Qwen3**：
[Qwen3-VL-235B-W8A8C16](modelslim/Qwen3-VL-235B-W8A8C16.md); 
[Qwen3-VL-235B-W8A8C16-Thinking](modelslim/Qwen3-VL-235B-W8A8C16.md);
[Qwen3-VL-32B-W8A8C16](./modelslim/Qwen3-VL-32B-W8A8C16.md);
[Qwen3-VL-32B-W8A8C8](./modelslim/Qwen3-VL-32B-W8A8C8.md);
[Qwen3-Coder-Next-W8A8C16](./modelslim/Qwen3-Coder-Next-W8A8C16.md)

**GLM5.0**：
[GLM5.0-W8A8C16](./modelslim/GLM5.0-W8A8C16.md)
[GLM5.0-W4A8C16](./modelslim/GLM5.0-W4A8C16.md)

**kimi2.5**:
[kimi2.5-W4A8C16](./modelslim/kimi2.5-W4A8C16.md)

**MiniMax-2.5**：
[MiniMax-2.5-W8A8C16](./modelslim/MiniMax-2.5-W8A8C16.md)

### 环境配置
#### 1. 按照下面脚本启动容器：
```bash
docker run -itd -u root \
--net=host \
--privileged \
--shm-size="500g" \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /etc/localtime:/etc/localtime \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /var/log/npu/:/usr/slog \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /sys/fs/cgroup:/sys/fs/cgroup:ro \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /efs_guiyang:/efs_guiyang \
-v /h240_models:/h240_models \
-v /data:/data \
-v /docker:/docker \
-v /mnt:/mnt \
-v /home/data:/home/data \
-v /home/data1:/home/data1 \
--entrypoint /bin/bash \
--name {container_name} registry-cbu.huawei.com/omniai_omniinfer_test/omniinfer-a3-arm:master-202512022014-vllm
```
#### 2. 安装modelslim
- 下载msmodelslim代码
`git clone https://gitcode.com/Ascend/msmodelslim.git`
- 进入到msmodelslim的目录并运行安装脚本
`bash install.sh`
#### 3. 量化
参照对应的[量化教程](#modelslim支持的权重)进行权重的量化


