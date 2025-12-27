import json
import os

GLOBAL_RANK_TABLE_KEY_ENV = 'GLOBAL_RANK_TABLE_FILE_PATH_KEY'


def get_global_node_ip_list():
    try:
        global_rank_table_path_key = os.getenv(GLOBAL_RANK_TABLE_KEY_ENV)
        global_rank_table_path = os.getenv(global_rank_table_path_key)
        with open(global_rank_table_path, 'r') as file:
            buf = file.read()
        rank_table = json.loads(buf)

        server_ip_list = []
        for group in rank_table['server_group_list']:
            for server in group['server_list']:
                server_ip_list.append(server['server_ip'])
        return server_ip_list
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    print(*get_global_node_ip_list(), sep=',')
