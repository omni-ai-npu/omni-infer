import json
import os

GLOBAL_RANK_TABLE_KEY_ENV = 'GLOBAL_RANK_TABLE_FILE_PATH_KEY'
PREFILL_POD_NUM_ENV= 'PREFILL_POD_NUM'


def get_prefill_head_ip_list():
    try:
        global_rank_table_path_key = os.getenv(GLOBAL_RANK_TABLE_KEY_ENV)
        global_rank_table_path = os.getenv(global_rank_table_path_key)
        with open(global_rank_table_path, 'r') as file:
            buf = file.read()
        rank_table = json.loads(buf)

        head_ip_list = []
        for group in rank_table['server_group_list']:
            head = group['server_list'][0]
            head_ip_list.append(head['server_ip'])

        prefill_instance_size = int(os.getenv(PREFILL_POD_NUM_ENV))
        return head_ip_list[:prefill_instance_size]
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    print(*get_prefill_head_ip_list(), sep=',')
