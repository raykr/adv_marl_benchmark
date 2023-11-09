import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor



# 最大并发执行的命令数量
max_concurrent_commands = 3

# 定义一个函数来执行单个命令
def execute_command(command):
    try:
        # 如果command为空白行或以#开头，则跳过
        if not command.strip() or command.strip().startswith("#"):
            return
        
        # 从 command 中提取 exp_name 后面的字符串 python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --exp_name mappo_smac_3m_gae_false --algo.use_gae False
        exp_name = command.split("--exp_name")[1].split("--")[0].strip()
        with open(f'logs/{exp_name}_stdout.log', 'w') as stdout_file, open(f'logs/{exp_name}_stderr.log', 'w') as stderr_file:
            print(f"Executing command: {command}")
            process = subprocess.Popen(
                command, shell=True, stdout=stdout_file, stderr=stderr_file, text=True
            )
            process.wait()  # 等待命令执行完成

        if process.returncode == 0:
            return f"Command '{command}' executed successfully"
        else:
            return f"Error executing command '{command}', please check the logs\n"
    except Exception as e:
        return f"Error executing command '{command}': {str(e)}\n"


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-s",
        "--script",
        type=str,
        required=True,
        help="You should provide a bash file with commands to execute.",
    )
    args, _ = parser.parse_known_args()
    args = vars(args)
    
    # 读取bash文件中的命令行
    with open(args["script"], 'r') as file:
        commands = file.readlines()

    # 使用ThreadPoolExecutor来控制并发执行
    with ThreadPoolExecutor(max_concurrent_commands) as executor:
        results = list(executor.map(execute_command, commands))

    # 打印执行结果
    for result in results:
        print(result)


if __name__ == '__main__':
    main()