import argparse
import subprocess


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        "-s",
        "--script",
        type=str,
        required=True,
        help="You should provide a bash file with commands to execute.",
    )
    args.add_argument(
        "-n",
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers to use for parallel execution.",
    )
    args = args.parse_args()
    
    # 执行command
    command = "cat " + args.script + " | xargs -I {} -P " + str(args.num_workers) + " bash -c '{} || echo \"{}\" >> errors.txt'"
    print(command)
    subprocess.run(command, shell=True)

