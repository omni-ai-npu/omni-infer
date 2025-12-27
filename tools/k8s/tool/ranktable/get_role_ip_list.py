import json
import os

GLOBAL_RANK_TABLE_KEY_ENV = 'GLOBAL_RANK_TABLE_FILE_PATH_KEY'
GROUP_ID_ENV = 'SERVER_GROUP_ID'


def get_role_ip_list():
    try:
        group_id = int(os.getenv(GROUP_ID_ENV))

        global_rank_table_path_key = os.getenv(GLOBAL_RANK_TABLE_KEY_ENV)
        global_rank_table_path = os.getenv(global_rank_table_path_key)
        with open(global_rank_table_path, 'r') as file:
            buf = file.read()
        rank_table = json.loads(buf)

        target_group = rank_table['server_group_list'][group_id]
        role_ip_list = []
        for server in target_group['server_list']:
            role_ip_list.append(server['server_ip'])
        return role_ip_list
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    print(*get_role_ip_list(), sep=',')
