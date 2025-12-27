import json
import os
import pathlib
import stat
import sys

GLOBAL_RANK_TABLE_KEY_ENV = 'GLOBAL_RANK_TABLE_FILE_PATH_KEY'
TRANS_GLOBAL_RANK_TABLE_FILE_PATH_ENV = 'TRANS_GLOBAL_RANK_TABLE_FILE_PATH'


def gen_global_rank_table():
    try:
        global_rank_table_path_key = os.getenv(GLOBAL_RANK_TABLE_KEY_ENV)
        global_rank_table_path = os.getenv(global_rank_table_path_key)
        with open(global_rank_table_path, 'r') as file:
            buf = file.read()
        global_rank_table = json.loads(buf)
        del global_rank_table['server_group_list'][0]
        global_rank_table['server_group_count'] = str(len(global_rank_table['server_group_list']))

        for group in global_rank_table['server_group_list']:
            group['group_id'] = str(int(group['group_id']) - 1)
        return global_rank_table
    except Exception as e:
        print(e)
        return None


def save_rank_table_to(rank_table):
    """
    将rank table信息存储到指定文件
    :param rank_table: rank table信息
    """
    file_path = os.getenv(TRANS_GLOBAL_RANK_TABLE_FILE_PATH_ENV)

    if not file_path:
        raise ValueError("Environment variable {} is not set".format(
            TRANS_GLOBAL_RANK_TABLE_FILE_PATH_ENV))
    
    # 创建父目录
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

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
    print("succeed to save global rank table in: " + file_path)


if __name__ == "__main__":
    save_rank_table_to(gen_global_rank_table())
