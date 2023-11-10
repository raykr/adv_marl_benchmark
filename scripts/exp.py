import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor


# 定义一个函数来执行单个命令
def execute_command(command):
    try:
        # 如果command为空白行或以#开头，则跳过
        if not command.strip() or command.strip().startswith("#"):
            return
        
        # 从 command 中提取 exp_name 后面的字符串 python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --exp_name mappo_smac_3m_gae_false --algo.use_gae False
        env_name = command.split("--env")[1].split("--")[0].strip()
        map_name = command.split("--env.map_name")[1].split("--")[0].strip()
        algo_name = command.split("--algo ")[1].split("--")[0].strip()
        exp_name = command.split("--exp_name")[1].split("--")[0].strip()
        trick_name = exp_name.replace(f"{algo_name}_{env_name}_{map_name}_", "")
        folder = f"{env_name}/{map_name}/{algo_name}/{trick_name}"
        os.makedirs(f"logs/{folder}", exist_ok=True)
        
        with open(f'logs/{folder}/stdout.log', 'w') as stdout_file, open(f'logs/{folder}/stderr.log', 'w') as stderr_file:
            print(f"Executing command: {command}")
            process = subprocess.Popen(
                command, shell=True, stdout=stdout_file, stderr=stderr_file, text=True
            )
            time.sleep(2)
            process.wait()  # 等待命令执行完成

        if process.returncode == 0:
            return f"Command '{command}' executed successfully"
        else:
            error_output = stderr_file.read()
            if "Error" in error_output:
                print(f"Error detected in stderr for command '{command}'. Restarting the process...")
                process.terminate() # 终止进程
                time.sleep(2) 
                return execute_command(command)
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
    parser.add_argument(
        "-n",
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers to use for parallel execution.",
    )
    args, _ = parser.parse_known_args()
    args = vars(args)
    
    # 读取bash文件中的命令行
    with open(args["script"], 'r') as file:
        commands = file.readlines()

    # 使用ThreadPoolExecutor来控制并发执行
    with ThreadPoolExecutor(args["num_workers"]) as executor:
        results = list(executor.map(execute_command, commands))

    # 打印执行结果
    for result in results:
        print(result)


if __name__ == '__main__':
    main()