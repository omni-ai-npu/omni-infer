import json
import os
import time

GLOBAL_RANK_TABLE_KEY_ENV = 'GLOBAL_RANK_TABLE_FILE_PATH_KEY'
POD_IP_ENV = 'POD_IP'


def wait_completed_global_rank_table():
    while True:
        try:
            pod_ip = os.getenv(POD_IP_ENV)

            global_rank_table_path_key = os.getenv(GLOBAL_RANK_TABLE_KEY_ENV)
            global_rank_table_path = os.getenv(global_rank_table_path_key)
            if not global_rank_table_path:
                print('read env \"{}\" failed'.format(global_rank_table_path_key))
            with open(global_rank_table_path, 'r') as file:
                buf = file.read()
            rank_table = json.loads(buf)

            if rank_table["status"] == "completed":
                server_group_list = rank_table['server_group_list']
                for group in server_group_list:
                    server_list = group["server_list"]
                    for i in range(len(server_list)):
                        if server_list[i]["server_ip"] == pod_ip:
                            return
                print("cannot find local ip in ranktable!")
            else:
                print("status of ranktable is not completed!")
        except Exception as e:
            print(e)
        time.sleep(1)


if __name__ == "__main__":
    wait_completed_global_rank_table()
