import json
import os

GLOBAL_RANK_TABLE_KEY_ENV = 'GLOBAL_RANK_TABLE_FILE_PATH_KEY'


def get_default_prefill_instance_size():
    try:
        global_rank_table_path_key = os.getenv(GLOBAL_RANK_TABLE_KEY_ENV)
        global_rank_table_path = os.getenv(global_rank_table_path_key)
        with open(global_rank_table_path, 'r') as file:
            buf = file.read()
        rank_table = json.loads(buf)

        # 默认场景认为是多P1D
        return len(rank_table['server_group_list']) - 1
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    print(get_default_prefill_instance_size())
