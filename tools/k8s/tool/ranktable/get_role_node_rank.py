import json
import os

GLOBAL_RANK_TABLE_KEY_ENV = 'GLOBAL_RANK_TABLE_FILE_PATH_KEY'
GROUP_ID_ENV = 'SERVER_GROUP_ID'
POD_IP_ENV = 'POD_IP'


def get_role_node_rank():
    try:
        group_id = int(os.getenv(GROUP_ID_ENV))
        pod_ip = os.getenv(POD_IP_ENV)

        global_rank_table_path_key = os.getenv(GLOBAL_RANK_TABLE_KEY_ENV)
        global_rank_table_path = os.getenv(global_rank_table_path_key)
        with open(global_rank_table_path, 'r') as file:
            buf = file.read()
        rank_table = json.loads(buf)

        node_rank = -1
        target_group = rank_table['server_group_list'][group_id]
        server_list = target_group["server_list"]
        for i in range(len(server_list)):
            node_rank += 1
            if server_list[i]["server_ip"] == pod_ip:
                return node_rank
        return -1
    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    print(get_role_node_rank())
