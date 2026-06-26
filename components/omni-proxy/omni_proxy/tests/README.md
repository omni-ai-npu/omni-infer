# proxy+vllm mock形态运行指南

测试用例运行前会启动vllm mock形态的后台进程和proxy进程，用例主要是测试端到端的请求发送和响应行为，发送request到proxy，proxy转发到vllm，vllm中模型执行过程被mock了，直接返回随机输出。

## 1. 前置操作

执行命令：

```shell
bash setup_vllm_mock.sh
```

## 2. 快速执行

执行命令：

```shell
pytest test_proxy.py -s -v
```

脚本里面随机分配可用端口，使用的端口号信息在当前目录的```shared_ports.json```中

## 3. 获取覆盖率报告

生成proxy覆盖率，执行命令：

```shell
bash gen_proxy_cov.sh
```

生成文件存放在当前目录的```proxy_report```目录下

覆盖率报告生成在当前目录的```htmlcov```目录下

## 4. 用例调试

test_proxy.py里面会通过fixture在用例执行前启动vllm_mock进程和proxy进程，调试时可以通过以下步骤快速调试单个用例：

1. 设置环境变量```export SKIP_FIXTURE=1```跳过每次执行pytest的setup步骤

2. 手动启动vllm_mock进程```python run_vllm_mock.py```，传参可以指定启动的prefill和decode节点数量，传入```stop```可以停止vllm进程

3. 手动启动proxy进程```python run_proxy.py```，传参可以指定启动的prefill和decode节点数量，传入```stop```可以停止proxy进程

4. 执行单个用例：```pytest test_proxy.py::test_xxx -s -v```

run_vllm_mock.py/run_proxy.py/test_xx.py脚本里面都有随机分配可用端口的功能，首先执行的脚本会随机端口号，并存储在当前目录的`shared_ports.json`中，后续其他脚本在使用时，假如指定的prefill和decode节点数量和json中对应port数量一致时，则会直接使用，否则会重新生成随机端口。

## 5. 日志路径

vllm_mock日志：当前目录下```prefill/decode_server_x.log```

nginx日志：```nginx_error.log```和```nginx_access.log```