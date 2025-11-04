# 支持 k8s ascend 集群部署

当前提供一个最小原型支持 omniinfer 在 k8s 上部署，作为演进的基础。

采用1P(POD) + 1D(POD) + 1C(POD)，即 Prefill实例、Decode实例、Global Proxy实例都采用1个POD部署。

## 一键部署

```bash
helm repo add omniinfer https://${HELM_REPO_URL}/omniai/omniinfer
helm install omniinfer # -f user-values-config.yaml # 用户修改部分配置
```

部署前，确保依赖已安装

卸载
```bash
helm uninstall omniinfer
```

## 安装依赖

1、部署好一个 k8s ascend 集群

包含组件kubectl命令行管理组件

可基于MindCluster，用Ascend Deployer工具批量安装集群调度组件[参考]

硬件建议规格

- ARM服务器：鲲鹏920芯片，64核以上配置
- NPU卡：昇腾910C，A3每节点16die
- 内存：256GB以上，支持NPU显存需求
- 存储：NVMe SSD用于模型存储

软件版本建议

- Kubernetes: 1.28+
- Ascend Device Plugin: v6.0.0+
- 操作系统: OpenEuler 22.03+（ARM版本）

配置好镜像仓库、网络代理等；

2、NFS 共享存储安装与配置：

```bash
IP_A=XXX # 机器A的IP（NFS 服务器）
mkdir -p /data/shared
vim /etc/exports # 添加`/data/shared *(rw,sync,no_root_squash,no_subtree_check)`
systemctl start nfs-server

# 在其他机器执行
yum install nfs-utils -y
mkdir -p /mnt/nfs_shared # 本地挂载点
mount -t nfs ${IP_A}:/data/shared /mnt/nfs_shared
df -h # 查看是否挂在成功
# showmount -e ${IP_A}
```

3、准备模型数据

下载到如 /data/models (默认路径)下；

```bash
# ls /data/models/
QwQ-32B
```

建议每台机器都通过NFS挂载，方便调度到任何一台机器上

```bash
mount -t nfs 7.150.8.27:/data/models /data/models
df -h
# Filesystem                  Size  Used Avail Use% Mounted on
# 7.150.8.27:/data/shared     3.0T  2.7T  147G  95% /mnt/nfs_shared
# 7.150.8.27:/data/models     3.0T  2.7T  147G  95% /data/models
```

4、安装 helm，执行 omniinfer 部署

下载helm [release版本](https://github.com/helm/helm/releases)，如v3.19.0 Linux arm64版本。

```bash
cd deploy/k8s/charts/omniinfer # Chart.yaml所在目录
helm install omniinfer . \
  --set mount.data.hostPath=/data \
  --set mount.model.path=/data/models \
  --set mount.log.path=/data/log_path
```

## 验证

验证1P1D分离的部署最小原型，部署QwQ-23B，打通流程，调用推理服务。

查看global-proxy-service推理服务入口：

```bash
# kubectl get svc
NAME                   TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
decode-service         ClusterIP      10.96.79.86     <none>        9100/TCP         3m42s
global-proxy-service   LoadBalancer   10.110.172.71   <pending>     8888:31778/TCP   3m41s
kubernetes             ClusterIP      10.96.0.1       <none>        443/TCP          7d5h
prefill-service        ClusterIP      10.104.21.50    <none>        9010/TCP         3m42s
```

调用服务：

```bash
# curl -X POST http://10.110.172.71:8888/v1/completions \
>   -H "Content-Type: application/json" \
>   -d '{
>         "model": "qwen",
>         "temperature": 0,
>         "max_tokens": 50,
>         "prompt": "how are you?"
>       }'
{"id":"cmpl-8a73aad9-0238-4e93-b2b7-0470722394c5","object":"text_completion","created":1761133871,"model":"qwen","choices":[{"index":0,"text":" I'm doing well, thank you for asking. How about you?\n\nI'm good too, thanks. Say, I have a question about something I read. It mentioned that the Earth's rotation is slowing down. Why is that happening? The Earth","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":4,"total_tokens":54,"completion_tokens":50,"prompt_tokens_details":null},"kv_transfer_params":null}
```

## 如何参与贡献

1、更多模型通过omniinfer k8s部署

最小原型支持QwQ-32B部署，更多模型的支持欢迎修改helm charts或提供模型部署配置参与贡献

用户helm配置部署验证，调测 helm charts

```bash
helm lint your-chart-directory/
helm template my-app . -f values.yaml  --dry-run --debug
helm template my-app . -f values.yaml  --show-only templates/prefill/prefill-deployment.yaml # 只查看一个文件的模板渲染结果
```

2、完善omniinfer k8s部署形态、安全、监控等支持

抛砖引玉，这里提供了1P1D1C的k8s最小原型, 打通omniinfer k8s组件部署流程，更多omniinfer组件还非k8s原生支持，如global proxy服务发现、P/D弹性扩缩、组件部署的最小安全权限、服务监控、压测等，欢迎参与完善

最小原型作为k8s部署验证演进的基础，其并不支持各故障场景的可靠性，k8s 控制器实现可靠性涉及组件k8s生命周期更多的精细控制，欢迎参与贡献

