import json
import os
import pathlib
import stat
import sys

CURRENT_STARTUP_ROLE_ENV = 'CURRENT_STARTUP_ROLE'
GLOBAL_RANK_TABLE_KEY_ENV = 'GLOBAL_RANK_TABLE_FILE_PATH_KEY'
GROUP_ID_ENV = 'SERVER_GROUP_ID'
LOCAL_RANK_TABLE_ENV = 'RANK_TABLE_FILE_PATH'
POD_IP_ENV = 'POD_IP'


def gen_rank_table():
    try:
        current_role = os.getenv(CURRENT_STARTUP_ROLE_ENV)
        group_id = int(os.getenv(GROUP_ID_ENV))
        pod_ip = os.getenv(POD_IP_ENV)

        global_rank_table_path_key = os.getenv(GLOBAL_RANK_TABLE_KEY_ENV)
        global_rank_table_path = os.getenv(global_rank_table_path_key)
        with open(global_rank_table_path, 'r') as file:
            buf = file.read()
        global_rank_table = json.loads(buf)
        group = global_rank_table['server_group_list'][group_id]

        rank_table = {'version': '1.0', 'group_id': str(group_id), 'server_count': '1', 'server_list': [], 'status': 'completed'}

        if current_role == "scheduler":
            for server in group['server_list']:
                if server['server_ip'] == pod_ip:
                    server['device'] = []
                    rank_table['server_list'] = [server]
                    break
        else:
            rank_table['server_list'] = group["server_list"]
            rank_table['server_count'] = str(len(group["server_list"]))
        return rank_table
    except Exception as e:
        print(e)
        return None


def save_rank_table_to(rank_table):
    """
    将rank table信息存储到指定文件
    :param rank_table: rank table信息
    """
    file_path = os.getenv(LOCAL_RANK_TABLE_ENV)

    # 同时挂载尚未生成的rank table目录可能创建同名空文件夹，检测并移除该目录
    if os.path.exists(file_path) and os.path.isdir(file_path):
        print("dir: {} already exists, remove it first.".format(file_path))
        dir_path = pathlib.Path(file_path)
        dir_path.rmdir()
    elif os.path.exists(file_path) and os.path.isfile(file_path):
        print("file: {} already exists, replace it with the latest version.".format(file_path))
        os.remove(file_path)

    # 读写权限
    flags = os.O_CREAT | os.O_WRONLY
    # 等同于chmod 644
    mode = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(file_path, flags, mode), 'w') as table_fp:
        json.dump(rank_table, table_fp, indent=4)
    sys.stdout.flush()
    print("succeed to save local rank table in: " + file_path)


if __name__ == "__main__":
    save_rank_table_to(gen_rank_table())
