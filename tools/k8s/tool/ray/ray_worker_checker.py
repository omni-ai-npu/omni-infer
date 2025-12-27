import json
import os
import ray
import time

GLOBAL_RANK_TABLE_KEY_ENV = 'GLOBAL_RANK_TABLE_FILE_PATH_KEY'
GROUP_ID_ENV = 'SERVER_GROUP_ID'


def wait_ray_workers_ready():
    target_count = get_role_node_size()

    ray.init(address='auto')
    while True:
        try:
            nodes = ray.nodes()
            if len(nodes) == target_count:
                print("ray workers are ready.")
                break
            else:
                print("ray workers not ready, check 5s later....")
                time.sleep(5)
        except Exception as e:
            print(e)


def get_role_node_size():
    try:
        group_id = int(os.getenv(GROUP_ID_ENV))

        global_rank_table_path_key = os.getenv(GLOBAL_RANK_TABLE_KEY_ENV)
        global_rank_table_path = os.getenv(global_rank_table_path_key)
        with open(global_rank_table_path, 'r') as file:
            buf = file.read()
        rank_table = json.loads(buf)

        target_group = rank_table['server_group_list'][group_id]
        return len(target_group['server_list'])
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    wait_ray_workers_ready()
