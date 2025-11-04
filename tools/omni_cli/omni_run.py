import os
import re
import shutil

INFO    = "\033[92m[INFO]\033[0m"      # green
WARNING = "\033[93m[WARNING]\033[0m"   # yellow
ERROR   = "\033[91m[ERROR]\033[0m"     # red

def create_run_parser(subparsers):
    # RUN command configuration
    run_parser = subparsers.add_parser("run", help="Run the omni service")
    run_parser.add_argument(
        'model_name',
        choices=['qwen', 'deepseek', 'kimi'],
        help='Type of model to deploy'
    )
    run_parser.add_argument(
        '--nodelist',
        default='127.0.0.1',
        type=str,
        help='List of node IPs, separated by commas, e.g.: ip1,ip2,ip3,ip4'
    )

    run_parser.add_argument(
        '--MODEL_PATH',
        required=True,
        type=str,
        help='Path to the model file, e.g.: /path/to/model'
    )

    run_parser.add_argument(
        '--config_path',
        default=None,
        type=str,
        help='Path to the configuration file, e.g.: /path/to/config.yaml'
    )

    run_parser.add_argument(
        '--LOG_PATH',
        default=None,
        type=str,
        help='Path to the log directory, e.g.: /path/to/logs'
    )

    run_parser.add_argument(
        '--ansible_ssh_private_key_file',
        required=True,
        type=str,
        help='Path to the SSH private key file, e.g.: /path/to/private_key'
    )
    run_parser.set_defaults(func=run)

def extract_node_count(template_filename, model_name):
    """
    Extract the number of nodes from the template filename.
    Supported filename formats: {model_name}_4node.yaml, {model_name}_8node.yaml, etc.

    Args:
        template_filename: The name of the template file
        model_name: The model name, default is "qwen"
    """

    # Use regex to match the numeric part, dynamically using the model name
    pattern = f"{model_name}_(\\d+)node"
    match = re.search(pattern, template_filename)
    if match:
        return int(match.group(1))

    # If regex fails, try other parsing methods
    # Remove the extension and split
    name_without_ext = template_filename.replace('.yaml', '').replace('.yml', '')
    parts = name_without_ext.split('_')

    for part in parts:
        if 'node' in part:
            # Extract the numeric part, e.g., "4node" -> 4
            num_str = part.replace('node', '')
            if num_str.isdigit():
                return int(num_str)

    # If none can be parsed, return default
    return 1

def select_template(node_count, available_templates, model_name):
    """
    Select the most suitable template.
    Strategy: Choose the largest template that is less than or equal to the node count.
    """
    # Create a hash table to store template info {node count: template filename}
    template_dict = {}

    available_file = os.listdir(available_templates)

    for template in available_file:
        # Parse node count from filename, e.g., "qwen_4node.yaml" -> 4
        size = extract_node_count(template, model_name)
        template_dict[size] = template

    # Get all available node count configurations and sort them
    available_sizes = sorted(template_dict.keys())

    # Find the largest template that is less than or equal to the node count
    selected_size = 1  # default to the smallest template

    for size in available_sizes:
        if size <= node_count:
            selected_size = size
        else:
            break

    # Get the corresponding template filename from the hash table
    return os.path.join(available_templates, template_dict[selected_size])

def replace_template(template_file, args, node_num):
    """
    Replace parameters in the template file with values from args

    Parameters:
        template_file (str): Path to the template file
        args (Namespace): A Namespace object parsed by argparse
        node_num (int): Number of nodes

    Returns:
        str: The path to the modified configuration file
    """
    # Convert Namespace to dictionary
    args_dict = vars(args)

    # Define the name of the copied file
    copy_filename = "server_profiles.yml"

    # Copy the template file to the current directory
    shutil.copy(template_file, copy_filename)

    # Read the file content as a string
    with open(copy_filename, "r", encoding="utf-8") as f:
        content = f.read()

    # Define the list of parameters to replace
    replace_keys = {
        # "docker_image_id",
        'ansible_ssh_private_key_file': 'TEMPLATE_SSH_PRIVATE_KEY_FILE',
        'MODEL_PATH': 'TEMPLATE_MODEL_PATH',
        'LOG_PATH': 'TEMPLATE_LOG_PATH',
        'container_name_prefix': 'TEMPLATE_CONTAINER_NAME_PREFIX'
    }

    # Replace parameters
    for key, value in replace_keys.items():
        if key in args_dict and args_dict[key] is not None:
            content = content.replace(value, str(args_dict[key]))

    # Replace node_list with ip1, ip2, ip3, etc.
    if "nodelist" in args_dict and args_dict["nodelist"] is not None:
        node_list = args_dict["nodelist"].split(',')
        ip_list = [f"ip{i + 1}" for i in range(len(node_list))]
        if len(ip_list) != node_num:
            raise ValueError(f"Node list length ({len(ip_list)}) does not match template requirement ({node_num})")
        for i, ip in enumerate(ip_list):
            content = content.replace(ip, node_list[i])

    # Write the modified content back to the file
    with open(copy_filename, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Configuration file has been modified and saved as: {copy_filename}")
    return copy_filename

def run(args):
    from omni_cli.main import run_docker_containers, omni_cli_start

    current_file_path = os.path.dirname(__file__)
    template_path = os.path.join(current_file_path, "configs")
    model_config_paths = [d for d in os.listdir(template_path) if os.path.isdir(os.path.join(template_path, d))]

    model_name = args.model_name
    # check whether the model is configured in the template path
    if model_name not in model_config_paths:
        raise ValueError(f"Model name {model_name} not found in template directory")

    model_config_path =  os.path.join(template_path, model_name)
    node_list = args.nodelist
    node_num = len(node_list.split(","))

    # Select the nearest node count configuration template
    template_file = select_template(node_num, model_config_path, model_name)
    template_file = replace_template(template_file, args, node_num)

    print(f"{INFO} start to run docker containers")
    try:
        run_docker_containers(inventory_path=template_file)
    except Exception as e:
        print(f"{ERROR} Error running docker containers: {e}")
        return
    print(f"{INFO} finish runing docker containers")


    print(f"{INFO} Running omni service with model {model_name} on nodes {node_list}")
    omni_cli_start(inventory_path = template_file,
                   skip_verify_config = True,
                   dev=False,
                   proxy_only = False)
    print(f"{INFO} Service started successfully")
