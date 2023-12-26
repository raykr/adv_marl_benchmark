import argparse
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument("-o", "--out", type=str, default="./out", help="out dir")
    args = parser.parse_args()
    
    # 执行command
    command = "cat " + args.script + " | parallel -j " + str(args.num_workers) + " 2>> " + args.out + "/logs/errors.txt"
    print(command)
    subprocess.run(command, shell=True)

