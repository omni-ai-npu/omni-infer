import stat
import os
import json
import time
import threading
from datetime import datetime

g_is_prefilling = False
g_node_number: str = ''

g_trace_enable = False
g_print_enable = False


class OmniLogger:
    """
    性能打点以及trace分析的工具类
    trace打点得到的json文件分析网站：https://ui.perfetto.dev/
    性能打点用来生成性能报告
    """

    TRACE_ATTR = 'trace_json_file'
    PRINT_ATTR = 'trace_print_file'

    _local = threading.local()

    def check_thread_file(self, file_attr: str):
        # 检查线程局部变量中是否有文件对象
        if file_attr == self.TRACE_ATTR and not hasattr(self._local, self.TRACE_ATTR):
            flags = os.O_CREAT | os.O_WRONLY
            mode = stat.S_IWUSR
            thread_id = threading.current_thread().ident
            trace_file = os.path.join(self.trace_path,
                                      self.name + "_trace_NODE_" + g_node_number + "_" + str(os.getpid()) + "_" + str(
                                          thread_id) + ".json")
            new_file = self.create_unique_file(trace_file)
            try:
                self._local.trace_json_file = os.fdopen(os.open(new_file, flags, mode), "w")
            except Exception as e:
                print("[OmniLogger] open trace file failed, %s", str(e))
            self._local.is_first = True
            self._local.trace_metrics = list()
        elif file_attr == self.PRINT_ATTR and not hasattr(self._local, self.PRINT_ATTR):
            flags = os.O_CREAT | os.O_WRONLY
            mode = stat.S_IWUSR
            thread_id = threading.current_thread().ident
            trace_file = os.path.join(self.trace_path,
                                      self.name + "_print_NODE_" + g_node_number + "_" + str(os.getpid()) + "_" + str(
                                          thread_id) + ".json")
            new_file = self.create_unique_file(trace_file)
            try:
                self._local.trace_print_file = os.fdopen(os.open(new_file, flags, mode), "w")
            except Exception as e:
                print("[OmniLogger] open print file failed, %s", str(e))

    def __init__(self, filepath: str = "", name=""):
        self.path = filepath
        self.name = name

        self.pid = os.getpid()
        self.tid = os.getpid()

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.trace_path = os.path.join(self.path, "trace")
        try:
            if not os.path.exists(self.trace_path):
                # print("make dir {self.trace_path}")
                os.mkdir(self.trace_path)
        except Exception as e:
            print("[OmniLogger] mkdir failed, err = %s", str(e))

    def add_json_begin(self):
        self.check_thread_file(self.TRACE_ATTR)
        self._local.trace_json_file.write('{"traceEvents":[\n')

    def add_event_begin(self, event_str: str, req_id: str, pid: int = 0, udf_time=None, args=None):
        """
        trace打点事件的开始打点
        参数：
            event_str: trace打点事件的名称
            req_id: 请求ID，必填，这里用于当做pid，将不通进程的同一个req_id事件放到同一个时间线上
            pid: trace的进程ID，在进行链路追踪时可为多个进程的trace事件设置同一pid。当前不使用
        """
        # print(f"add_event_begin, event_str = {event_str}")
        self.check_thread_file(self.TRACE_ATTR)
        metric = dict()
        metric["name"] = event_str
        metric["ph"] = "B"
        # metric["pid"] = self.pid if pid == 0 else pid
        metric["pid"] = req_id
        if udf_time:
            metric["ts"] = udf_time  # ts的时间精度为us
        else:
            metric["ts"] = round(time.time_ns() / (10 ** 3))
        if args:
            metric["args"] = args
        self._local.trace_metrics.append(metric)
        self.output()
        # if len(self._local.trace_metrics) == 1000:
        #     self.output()

    def add_event_end(self, event_str: str, req_id: str, pid: int = 0, udf_time=None, args=None):
        """
        trace打点事件的结束打点
        参数：
            event_str: trace打点事件的名称
            req_id: 请求ID，必填，这里用于当做pid，将不通进程的同一个req_id事件放到同一个时间线上
            pid: trace的进程ID，在进行链路追踪时可为多个进程的trace事件设置同一pid。当前不使用
        """
        # print(f"add_event_end, event_str = {event_str}")
        self.check_thread_file(self.TRACE_ATTR)
        metric = dict()
        metric["name"] = event_str
        metric["ph"] = "E"
        # metric["pid"] = self.pid if pid == 0 else pid
        metric["pid"] = req_id
        if udf_time:
            metric["ts"] = udf_time
        else:
            metric["ts"] = round(time.time_ns() / (10 ** 3))
        if args:
            metric["args"] = args
        self._local.trace_metrics.append(metric)
        self.output()
        # if len(self._local.trace_metrics) == 1000:
        #     self.output()

    def add_event_tag(self, event_str: str, req_id: str, pid: int = 0, udf_time=None, args=None):
        """
        trace打点事件的瞬时打点
        参数：
            event_str: trace打点事件的名称
            req_id: 请求ID，必填，这里用于当做pid，将不通进程的同一个req_id事件放到同一个时间线上
            pid: trace的进程ID，在进行链路追踪时可为多个进程的trace事件设置同一pid。当前不使用
        """
        # print(f"add_event_tag, event_str = {event_str}")
        self.check_thread_file(self.TRACE_ATTR)
        metric = dict()
        metric["name"] = event_str
        metric["ph"] = "i"
        # metric["pid"] = self.pid if pid == 0 else pid
        metric["pid"] = req_id
        metric["s"] = "p"
        if udf_time:
            metric["ts"] = udf_time
        else:
            metric["ts"] = round(time.time_ns() / (10 ** 3))
        if args:
            metric["args"] = args
        self._local.trace_metrics.append(metric)
        self.output()
        # if len(self._local.trace_metrics) == 1000:
        #     self.output()

    def create_unique_file(self, original_name):
        base_name, extension = os.path.splitext(original_name)
        counter = 1

        while os.path.exists(original_name):
            original_name = f"{base_name}_{counter}{extension}"
            counter += 1

        return original_name

    def output(self):
        """
        将性能打点和trace打点输出到文件
        """
        if len(self._local.trace_metrics):
            for item in self._local.trace_metrics:
                json_str = json.dumps(item, separators=(',', ':'))
                if not self._local.is_first:
                    json_str = "\n" + json_str
                self._local.is_first = False
                self._local.trace_json_file.write(json_str)
                self._local.trace_json_file.flush()
            self._local.trace_metrics = []

    def finalize(self):
        self.output()
        self._local.trace_json_file.write('\n],"displayTimeUnit": "us"}')
        self._local.trace_json_file.close()

    def add_print(self, *args, **kwargs):
        self.check_thread_file(self.PRINT_ATTR)
        print(*args, **kwargs, file=self._local.trace_print_file, flush=True)

    def add_print_timestamp(self, req_id, action: str):
        self.check_thread_file(self.PRINT_ATTR)
        print(f'profile REQ_ID[{req_id}] action:{action}.Timestamp {time.time()}', file=self._local.trace_print_file,
              flush=True)


# OMNILOGGER_ARGS=/dev/shm,P0,01
# arg[0]: 日志文件保存路径，比如: /dev/shm
# arg[1]: vllm节点角色和编号，P0/D10
# arg[2]: 按照对应的位置是否为1表示对应的打点是否开启，[0]: trace, [1]: print

g_trace: OmniLogger = None

if os.environ.get("OMNILOGGER_ARGS") != None:
    if g_trace == None:
        env = os.environ.get("OMNILOGGER_ARGS")
        args = env.split(',')
        if args[1][0] == "P":
            g_is_prefilling = True
        g_node_number = args[1]
        g_trace_enable = args[2][0] == "1"
        g_print_enable = args[2][1] == "1"
        g_trace = OmniLogger(filepath=args[0], name="omni")


def omni_logger_enable() -> bool:
    return g_trace != None


def omni_logger_trace_enable() -> bool:
    return g_trace_enable


def omni_logger_print_enable() -> bool:
    return g_print_enable


def omni_logger_add_event_begin(req_id: str, event_str: str, udf_time=None, args=None, use_role=False):
    # print("==== CUDA omni_logger_add_event_begin ====")
    if g_trace and g_trace_enable:
        if use_role:
            event_str = ("P_" if g_is_prefilling else "D_") + event_str
        g_trace.add_event_begin(event_str=event_str, req_id=req_id, udf_time=udf_time, args=args)


def omni_logger_add_event_end(req_id: str, event_str: str, udf_time=None, args=None, use_role=False):
    # print("==== CUDA omni_logger_add_event_end ====")
    if g_trace and g_trace_enable:
        if use_role:
            event_str = ("P_" if g_is_prefilling else "D_") + event_str
        g_trace.add_event_end(event_str=event_str, req_id=req_id, udf_time=udf_time, args=args)


def omni_logger_add_event(req_id: str, event_str: str, udf_time=None, args=None, use_role=False):
    if g_trace and g_trace_enable:
        if use_role:
            event_str = ("P_" if g_is_prefilling else "D_") + event_str
        g_trace.add_event_tag(event_str=event_str, req_id=req_id, udf_time=udf_time, args=args)


def omni_logger_is_prefilling() -> bool:
    return g_is_prefilling


def omni_logger_print(*args, **kwargs):
    if g_trace and g_print_enable:
        g_trace.add_print(*args, **kwargs)


def omni_logger_print_timestamp(req_id, action: str):
    if g_trace and g_print_enable:
        g_trace.add_print_timestamp(req_id=req_id, action=action)
