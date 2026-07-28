# 基于KubeInfer的k8s环境服务拉起介绍

## 环境准备
当前拉起方式需要在每个节点机器上部署相关k8s环境：k8s、KuberInfer、helm、Volcano等;

我们提供了支持一键部署基于KubeInfer的k8s环境的脚本，具体参考`/omniinfer/tools/install/README.md`

## 脚本介绍

```bash
└── k8s
    ├── config # 环境变量相关配置脚本
    ├── role # 不同role的起服务脚本
    ├── tool # 部分工具脚本
    ├── health.sh # 健康检查脚本
    ├── pre_stop.sh # 优雅退出脚本
    ├── start.sh # 服务启动脚本
    ├── omni_kubeinfer.yaml # k8s服务所需的deployment
    └── README.md
```

<!-- ## omni_kubeinfer.yaml部分字段说明

该文件定义了KubeInfer和K8s需要的配置信息，文件参数配置说明如下： -->

## 操作步骤
### 修改 omni_kubeinfer.yaml 配置
在omni_kubeinfer.yaml中，每个`recoveryPolicy`字段对应一种Role，顺序分别为C节点、Prefill节点和Decode节点。k8s会自动调度机器，因此需要在修改每个`recoveryPolicy`的下一级属性字段，且尽量保持一致。

主要的修改的配置说明如下：

- `replicas`: 副本数，表示当前role下面的pod副本数量，和`sever_list`的`length`相关。如多机组D，`replicas`设为2。
- `nodeSelector`: 用于指定机器node节点，Volcano会自动选择指定label的node进行调度。`nodes-name-prefix`为指定的label。
- `image`: 用于指定服务所需的对应镜像。
- `imagePullPolicy`: 镜像拉取策略。
- `huawei.com/ascend-1980`：用于指定NPU的数目。
- `volumeMounts`：该字段下的`mountPath`对应挂载到容器内的路径，`volumes`和`volumeMounts`之间的name必须保持一一对应。只读文件可设为`readOnly`.
- `volumes`：该字段下的`hostPath`对应挂载到容器内的宿主机路径。

### 服务拉起配置的传入
k8s服务的拉起主要通过`start.sh`执行，拉起的环境变量配置也由该脚本传入。大部分环境变量都在`omniinfer/tools/k8s/config/env`路径下的脚本文件赋默认值，传入参数时，会覆盖这些默认值。例如：
```bash
bash /workspace/omniinfer/tools/k8s/start.sh --port=8080 --served-model-name=deepseek --enable-logging-config=0 --model-path=/sfs/model/DeepSeek-R1-w8a8-fusion --gpu-util=0.85 --max-model-len=65536
```
### 执行服务拉起命令
修改好omni_kubeinfer.yaml配置文件后，就可以执行k8s的一些命令来拉起服务和查看运行状态。常用命令如下：
```shell
kubectl get pods # 查看pod状态

kubectl get nodes # 查看node状态

kubectl apply -f omni_kubeinfer.yaml # 应用/更新资源，服务拉起命令

kubectl delete -f omni_kubeinfer.yaml # 删除当前节点

kubectl logs <pod-name> # 查看Pod实时日志（常用于排查应用内部问题核心）

kubectl describe pod <pod-name> # 查看资源详细信息

kubectl exec -it <pod-name> -n <namespace> -- bash # 进入Pod内部执行命令或调试。
```

查看pod状态，所以pod的`READY`都显示为`1/1`时，说明服务正常拉起。