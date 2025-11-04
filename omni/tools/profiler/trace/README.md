# Trace接入Jaeger使用文档


为了提升OmniInfer在可视化监控、运维及性能分析方面的能力，现将**vLLM**和**Omni Proxy**的trace数据接入Jaeger，实现分布式链路追踪。通过此工具，开发者与运维可直观观测、分析模型推理过程中的各环节性能瓶颈与异常。
- **可视化链路追踪**：通过Jaeger UI查看每个请求的全链路trace，包含各阶段耗时、依赖关系、上下游调用等信息。
- **性能瓶颈定位**：通过span耗时分布，快速定位慢点、异常点。
- **运维监控**：方案一离线日志导入Jaeger支持对trace的统计分析。
- **历史数据分析**：可导出trace数据进行离线分析，或通过脚本批量导入历史trace进行复盘。

目前trace设计只支持vLLM和Omni Proxy，将离线trace打点日志导入Jaeger，后续会开发Jaeger自动抓取trace。


## 1. 环境准备与搭建

### 1.1 部署Jaeger和OpenTelemetry Collector
vllm trace导入Jaeger的框架示意图为：

```
                                                                                   
+-------------------+         4317(gRPC)        +---------------------------+   4317(OTLP gRPC)            +----------------------+
|      vLLM         | ========================> |   OTEL Collector          | ===========================> |      Jaeger          |
| (产出OTLP Trace)  |       或 4318(HTTP)       | (otel/opentelemetry-...)  |       或14250(gRPC)          | (jaegertracing/all-) |
+-------------------+                           +---------------------------+                              +----------------------+
                                                                                                                     ↑
                                                                                                               16686 (Web UI)
                                                                                                                     ↑
                                                                                                             用户浏览器访问查看trace
```
- **vLLM**：产出OTLP trace数据（OpenTelemetry格式）。
- **OpenTelemetry Collector**：接收vLLM发来的trace数据，进行处理和转发。
- **Jaeger**：存储和展示trace数据，提供UI分析界面。

#### 下载镜像
```bash
docker pull otel/opentelemetry-collector-contrib:0.103.1
docker pull jaegertracing/all-in-one:1.56
```
（如果服务器拉取镜像失败，可以尝试本地下载然后传至远程服务器）

#### OpenTelemetry Collector 端口对接配置

配置文件：`otel.yml`

```yaml
receivers:
  otlp:
    protocols:
      grpc:                            # 4317
      http:  
        endpoint: 0.0.0.0:4318         # 4318

processors:
  batch:

exporters:
  otlp:                       # 用 otlp 取代 jaeger
    endpoint: jaeger:4317     # Jaeger 自带的 OTLP gRPC 端口
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp]
```

#### 一键启动容器


配置文件：`docker-compose.yml`

```yaml
services:
  jaeger:
    image: jaegertracing/all-in-one:1.56
    ports:
      - "16686:16686"   # Jaeger UI
      - "14250:14250"   # gRPC 接收
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.103.1
    command: ["--config=/etc/otel.yml"]
    volumes:
      - ./otel.yml:/etc/otel.yml:ro
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
    depends_on:
      - jaeger
```

将`otel.yml`和`docker-compose.yml`两个文件放在同一目录下，然后在该目录下执行：
```bash
docker compose up -d
```
就能一次性启动两个容器（Jaeger + OTel Collector）。打开网址``http://localhost:16686`` 就可以正常连接显示Jaeger UI页面。


### 1.2 服务器集群时钟同步
分布式trace追踪需要确保机器时钟同步，详细步骤说明可以参考这篇wiki：https://codehub-g.huawei.com/DataScience/omni_infer/wiki?categoryId=221789&sn=WIKI202506247247579&title=6--%E9%98%B2%E7%81%AB%E5%A2%99%E6%94%BE%E8%A1%8C-NTP-UDP-123- 
这里我们使用chrony 配置时间同步：
- 安装 chrony（如果未安装）
```bash
yum install -y chrony    # CentOS 7/8
dnf install -y chrony    # CentOS 8+ / Rocky Linux
```
- 配置 chrony： 编辑 /etc/chrony.conf，指定 NTP 服务器
```bash
vim /etc/chrony.conf
```
推荐配置（适用于内网/公网/自定义服务器）
```bash
# NTP Server Configuration（本次方法使用这个）
 pool 172.19.1.63 iburst
```
- 启动并启用 chrony
```bash
systemctl enable --now chronyd
systemctl stop chronyd
systemctl start chronyd
systemctl status chronyd
```
- 检查同步状态
```bash
chronyc sources -v      # 查看同步的 NTP 源
chronyc tracking        # 查看当前同步状态
```
- 多窗口执行 date 命令，可快速验证多台服务器的时钟对齐。比如
```bash
date -u +"%Y-%m-%d %H:%M:%S.%6N"
```


## 2. 方案一：离线日志导入Jaeger

通过`log_to_jaeger.py`脚本，把vLLM和Omni Proxy的trace打点log转化为Jaeger可识别的trace数据（JSON格式）。（`注意必须确保服务器集群时钟同步后，再去打点生成日志。`）
#### 开启trace打点
开启vllm trace打点：在``pd_run.sh``里导入以下变量路径，打点log路径在``TRACE_OUTPUT_DIRECTORY``
```bash
export PROFILING_NAMELIST = {project_root}/omni_infer/omni/tools/profiler/assets/omnilogger_namelist
export TRACE_OUTPUT_DIRECTORY=/your/custom/path
```

开启Omni Proxy trace打点: 在``omni_infer_server_template.yml``里找到``run_proxy_cmd``，确保``omni_proxy.sh``命令的log等级为info，打点log路径在``LOG_PATH``
```bash
bash omni_proxy.sh \
  --log-level info \
```
#### 从各个服务器收集log到本地：

`server_list.txt`保存服务器IP:

```
10.11.123.1
10.11.123.2
10.11.123.3
10.11.123.4
```

 `collect_logs.sh`将所有服务器的打点log复制到本地目录`your_log_directory`，注意key.pem需要换成自己的密钥

```bash
#!/bin/bash
# Usage: ./collect_logs.sh server_list.txt TRACE_OUTPUT_DIRECTORY LOG_PATH your_log_directory

SERVER_LIST="$1"
REMOTE_FOLDER="$2"
PROXY_FOLDER="$3"
TARGET_FOLDER="$4"

mkdir $TARGET_FOLDER

for IP in $(cat "$SERVER_LIST"); do
    echo "Collecting logs from $IP..."
    scp -i key.pem -r "root@$IP:$REMOTE_FOLDER" "./logs_$IP"
    mv "./logs_$IP" $TARGET_FOLDER

    # copy logs in proxy
    if ssh -i key.pem root@$IP "test -f '$PROXY_FOLDER/nginx_error.log'"; then
        echo "nginx_error.log found on $IP, copying..."
        scp -i key.pem "root@$IP:$PROXY_FOLDER/nginx_error.log" "$TARGET_FOLDER/nginx_${IP}.log"
    fi
done

```
#### 将log日志转为Jaeger可识别的JSON文件

每个请求的打点日志均可解析为 Jaeger 可识别的 JSON 文件，并支持对所有请求的 Trace 进行统计分析（可排除前后各 num 个请求以提高统计准确性）。统计内容包括 Request Average Trace，Request TP90 Percentile Trace，Request TP95 Percentile Trace，Request TP99 Percentile Trace。

```bash
# 默认配置下，--skip-req_num 参数为 1，--filter-reqids 为空。脚本会自动排除首尾各一个请求，对其余请求进行 Trace 统计，并为每个请求生成 Trace 文件。
python log_to_jaeger.py your_log_directory trace.json 

# 当 --skip-req_num 参数为 num 时，脚本将排除最前面和最后面各 num 个请求，对剩余请求进行 Trace 统计，并为每个请求生成 Trace 文件。
python log_to_jaeger.py /path/to/log_dir trace.json --skip-req_num num

# 当指定 --filter-reqids 参数为 reqid1,reqid2,reqid3 时，脚本仍默认排除首尾各一个请求，对剩余请求进行 Trace 统计，并仅为 request_id 为 reqid1、reqid2、reqid3 的请求生成 Trace 文件。
python log_to_jaeger.py /path/to/log_dir trace.json --filter-reqids reqid1,reqid2,reqid3

# 当同时指定 --skip-req_num 和 --filter-reqids 参数时，脚本会排除前面和后面各 num 个请求然后对其余请求进行trace统计，仅对指定的请求生成 trace 文件。
python log_to_jaeger.py /path/to/log_dir trace.json --skip-req_num num --filter-reqids reqid1,reqid2,reqid3
```

#### 将解析的JSON文件导入Jaeger UI 
打开网页``http://localhost:16686``，显示Jaeger UI界面正常连接，就可以直接导入json文件到Jaeger UI进行可视化分析，查看请求完整的生命周期。这里如果您使用内网，可以直接使用已经搭建好的UI``http://7.150.8.141:16686``
- **优点**：无需步骤1中Jaeger及OpenTelemetry相关环境部署，仅需本地处理和导入，适合离线分析和调试。
- **缺点**：不能实时监控，流程相对繁琐，不适合生产环境的实时观测。

## 3 方案二：Jaeger自动抓取trace(开发中)

- vLLM配置好OpenTelemetry SDK，通过非侵入式方式在vllm内部创建span，将OTLP trace通过OTLP协议自动发送至本地或远端的OTEL Collector（端口4317或4318）。Collector处理后，将trace转发至Jaeger服务（端口4317）。
- 用户可通过Jaeger UI（http://localhost:16686）实时搜索、查看、分析vLLM的全链路追踪信息。
- **优点**：支持实时监控和分析，适合生产环境，便于自动化运维与报警。
- **缺点**：需额外维护Collector与Jaeger实例，会影响推理性能。