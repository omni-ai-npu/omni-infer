## 云道脚本
云道拉起脚本，基于omni cli的拉起脚本，自适应拉起云道服务
### 功能简介
- ip替换
```
识别云道ip并替换server_profiles.yml中的对应ip
```
- port替换
```
识别云道proxy port并替换server_profiles.yml中的对应proxy port
```
- 路径替换
```
修改模型路径、日志路径、ranktable路径为云道路径
```
- 脚本自动生成
```
依赖云道镜像中的omni cli(默认预安装)，生成所有节点对应的拉起服务脚本，然后各节点运行自己对应的脚本来拉起服务
```
 ### 配置
 - 云道启动命令，为：bash /path_to_script/cloud_run_v0.sh
 - 云道需要配置的环境变量见ENV.xlsx