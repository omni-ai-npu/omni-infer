#!/bin/bash


#获取当前执行路径
cur_path=$(cd "$(dirname "$0")" && pwd)

usage() {
    echo "用法: $0 [选项] [参数...]"
    echo
    echo "选项:"
    echo "  -h, -help                                        显示此帮助信息"
    echo "  -log_path, -lp <log path>                        指定需要轮转的日志路径"
    echo "  -rotate_count, -rc <0,1,2,3...>                  指定需要保留的历史日志数目，0代表不保存"
    echo "  -raylet_size, -rs <10,10k,10M,10G..>             指定raylet gcs日志切分大小，默认单位为k"
    echo "  -other_size, -os <10,10k,10M,10G..>              指定除raylet gcs之外的日志切分大小，默认单位为k"
    echo "  -frequency, -f <minutes,hourly,daily>            指定日志轮转频率，支持分钟级，小时级和天级"
    echo "  -cron_minute, -cron_minute <0,1,2,3...60>        指定分钟级频率的轮转间隔，或者小时级，天级下的运行时间(分)"
    echo "  -cron_hour, -ch <0,1,2,3...23>                   指定小时级频率的轮转间隔，或者天级下的运行时间(小时)"
    echo
    echo "示例:"
    echo "  $0 -lp /tmp/ray/session_latest/logs/ -rc 0 -rs 100M -os 10M -f hourly -cn 10 -ch 1"
    echo "  $0 -help"
}

# 解析输入参数
while [ -n "$1" ]; do
  case $1 in
    -help) shift
          usage
          ;;
    -h) shift
          usage
          ;;
    -log_path) shift
          ray_log_path=$1
          ;;
    -lp) shift
          ray_log_path=$1
          ;;
    -rotate_count) shift
          rotate_num=$1
          ;;
    -rc)   shift
          rotate_num=$1
          ;;
    -raylet_size) shift
          raylet_rotate_size=$1
          ;;
    -rs) shift
          raylet_rotate_size=$1
          ;;
    -other_size) shift
          others_rotate_size=$1
          ;;
    -os) shift
        others_rotate_size=$1
        ;;
    -frequency) shift
        if [ $1 == "daily" ] || [ $1 == "hourly" ] || [ $1 == "minutes" ]; then
            frequency=$1
        else
            echo "轮转频率只支持minutes，hourly，daily，$1 未识别，使用默认daily"
            frequency="daily"
        fi
         ;;
    -f) shift
        if [ $1 == "daily" ] || [ $1 == "hourly" ] || [ $1 == "minutes" ]; then
            frequency=$1
        else
            echo "轮转频率只支持minutes，hourly，daily，$1 未识别，使用默认daily"
            frequency="daily"
        fi
         ;;
    -cron_minute) shift
        cron_minute=$1
        ;;
    -cm) shift
        cron_minute=$1
        ;;
    -cron_hour) shift
        cron_hour=$1
        ;;
    -ch) shift
        cron_hour=$1
        ;;
    *) echo "invalid option word: $1 , skip ~"

  esac
  shift
done

#自定义ray log路径
if [ -z "$ray_log_path" ]; then
    ray_log_path="/tmp/ray/session_*/logs"
fi

#自定义历史日志保存数目
if [ -z "$rotate_num" ]; then
    rotate_num=0
fi

#自定义raylet,gcs log切分大小
if [ -z "$raylet_rotate_size" ]; then
    raylet_rotate_size=100M
fi

#自定义除raylet,gcs log之外的所有log的切分大小
if [ -z "$others_rotate_size" ]; then
    others_rotate_size=10M
fi

if [ -z "$frequency" ]; then
    frequency="hourly"
fi

#检查是否安装logrotate程序
logrotate_exist=$(ls /usr/sbin | grep logrotate | wc -l)
if [ "$logrotate_exist" -lt 1 ]; then
    echo "没有发现logrotate程序，请先正确安装logrotate后再次重试！"
    exit 1
fi

#创建ray log的logrotate 配置文件
mkdir -p /etc/logrotate.d/ray
#设置raylet gcs等较大日志切分配置
sed "s|RAY_LOG_PATH|$ray_log_path|g; s|ROTATE_NUM|$rotate_num|g; s|ROTATE_SIZE|$raylet_rotate_size|g" $cur_path/raylet-gcs.conf > /etc/logrotate.d/ray/raylet-gcs.conf
#设置其他ray日志切分配置
sed "s|RAY_LOG_PATH|$ray_log_path|g; s|ROTATE_NUM|$rotate_num|g; s|ROTATE_SIZE|$others_rotate_size|g" $cur_path/ray-others.conf > /etc/logrotate.d/ray/ray-others.conf

#检查是否开启crontab服务
cron_proc_exist=$(ps -aux | grep crontab | wc -l)
if [ "$cron_proc_exist" -lt 2 ]; then
    echo "crontab service not start up, try to start crontab ……"
    /usr/sbin/crond
    sleep 1
fi
#再次检查crontab服务是否启动
cron_proc_exist=$(ps -aux | grep crond | wc -l)
if [ "$cron_proc_exist" -lt 2 ]; then
    echo "crontab service启动失败，将无法使用logrotate进行日志切分！"
    exit 1
fi
#删除已有logrotate的crontab任务
rm -rf /etc/cron.daily/logrotate
rm -rf /etc/cron.hourly/logrotate
rm -rf /etc/cron.monthly/logrotate
rm -rf /etc/cron.weekly/logrotate

crontab -l | awk '$7 !~/logrotate/' > /tmp/cron_clean
crontab /tmp/cron_clean

#自定义logrotate的crontab任务执行频率时间，单位分钟
cron_reg="30 2 * * *"

#编辑天级别任务
if [ $frequency == "daily" ]; then
    if [ -z "$cron_minute" ]; then
        cron_minute=30
    fi
    if [ -z "$cron_hour" ]; then
        cron_hour=2
    fi
    cron_reg="$cron_minute $cron_hour * * *"
    sed -i '13d' /etc/logrotate.d/ray/raylet-gcs.conf
    sed -i '31d' /etc/logrotate.d/ray/ray-others.conf
fi
#编辑小时级别任务
if [ $frequency == "hourly" ]; then
    if [ -z "$cron_minute" ]; then
        cron_minute=30
    fi
    if [ -z "$cron_hour" ]; then
        cron_hour=1
    fi
    cron_reg="$cron_minute */$cron_hour * * *"
fi
#编辑分钟级别任务
if [ $frequency == "minutes" ]; then
    if [ -z "$cron_minute" ]; then
        cron_minute=10
    fi
    cron_reg="*/$cron_minute * * * *"
fi

#首次运行，生成stub state file
/usr/sbin/logrotate -v /etc/logrotate.d/ray/*

#设置crontab任务
echo "$cron_reg  bash $cur_path/set_logrotate_cron.sh"  | crontab -u root -

echo "ray日志的logrotate切分配置设置成功，后续可在$ray_log_path/log_rotate.log中查看运行记录！"

exit 0