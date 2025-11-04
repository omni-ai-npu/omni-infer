# ems
## 简介
弹性内存存储（Elastic Memory Service，EMS）是一种以DRAM内存（动态随机存取存储器）为主要存储介质的云基础设施服务，为LLM推理提供缓存和推理加速。EMS实现AI服务器的分布式内存池化管理，将LLM推理场景下多轮对话及公共前缀等历史KVCache缓存到EMS内存存储中，通过以存代算，减少了冗余计算，提升推理吞吐量，大幅节省AI推理算力资源，同时可降低推理首Token时延（Time To First Token，TTFT），提升LLM推理对话体验。
更多介绍请查看[华为云-弹性内存存储 EMS](https://support.huaweicloud.com/productdesc-ems/ems_01_0100.html)。

## 在omni-infer中使用
### 环境准备
1. 部署好ems集群（参照[华为云-弹性内存存储 EMS](https://support.huaweicloud.com/productdesc-ems/ems_01_0100.html)申请公测)。
2. 确保将宿主机EMS服务端容器共享的unix domain socket目录"/mnt/paas/kubernetes/kubelet/ems"，通过增加负载配置文件hostPath项，将目录映射到推理容器目录："/dev/shm/ems"；同时在推理容器内，运行服务的用户能够读写该文件夹及其文件。
3. 在推理容器内运行pip install ems-.*linux_aarch64.whl命令执行安装对应版本的python sdk。混部场景下，KV_CONNECTOR设置为CcConnector。PD分离场景下，Prefill节点的KV_CONNECTOR设置为EmsConnector。
设置环境变量
在`tools/ansible/template/omni_infer_server_template.yml` 中的`run_vllm_server_prefill_cmd`下添加EMS相关环境变量。

| 变量名称 | 变量类型 | 描述 |
|----|----|-----|
| MODEL_ID | string | 参数解释：<br>唯一标识当前推理服务使用的推理模型ID。<br><br>约束限制：<br>1. 1～512个字符，支持数字、小写字母、“.”、“-”、“_”。<br>2. 需要保证全局唯一。<br><br>默认取值：<br>“cc_kvstore@_@ds_default_ns_001”。
| ACCELERATE_ID | string | 参数解释：<br>业务访问内存池身份凭证，由用户自行指定并保证唯一性，在需要进行业务多租隔离场景使用。<br><br>约束限制：<br>1. 1～512个字符，支持数字、小写字母、“.”、“-”、“_”。<br>2. 需要保证全局唯一。<br><br>默认取值：<br>“access_id”。
| EMS_ENABLE_WRITE_RCACHE   | int     | 参数解释：<br>控制是否将本次写入保存为本地读缓存。<br><br>约束限制：必须为数字。<br><br>取值范围：<br>“1”：保存。<br>其他：不保存。<br><br>默认取值：<br>“1”。 |
| EMS_ENABLE_READ_LOCAL_ONLY| int     | 参数解释：<br>控制是否只读本地缓存。<br><br>约束限制：必须为数字。<br><br>取值范围：<br>“1”：仅读取本地缓存（不从其他节点读取）。<br>其他：优先读本地缓存、未命中则从其他节点读取。<br><br>默认取值：<br>“0”。  |
| EMS_TIMEOUT        | int      | 参数解释：<br>定义请求超时时间（单位：毫秒）。<br><br>取值范围：<br>大于等于0。<br><br>默认取值：<br>“5000”。 |