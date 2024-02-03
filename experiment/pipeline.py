

import argparse
import os
import subprocess


def execute_command(command, capture_output=False):
    print(f"\033[32m==> {command}\033[0m")
    process = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
    # print(f"\033[31m{process}\033[0m")
    return process


def train(args):
    process = execute_command(f"python generate.py train {args.extra}", capture_output=True)
    execute_command(process.stdout)


def eval_all(args):
    process = execute_command(f"python generate.py eval {args.extra}", capture_output=True)
    execute_command(process.stdout)


def eval_stage_1(args):
    process = execute_command(f"python generate.py eval {args.extra} --stage 1", capture_output=True)
    execute_command(process.stdout)


def eval_stage_2(args):
    process = execute_command(f"python generate.py eval {args.extra} --stage 2", capture_output=True)
    execute_command(process.stdout)


def export(args):
    execute_command(f"python export.py {args.extra}")


def plot(args):
    execute_command(f"python plot.py {args.extra}")


def rsync(args):
    # 读取.env中的配置项
    from dotenv import load_dotenv

    # 加载.env文件
    load_dotenv()

    # 读取环境变量
    remote_server = os.getenv('REMOTE_SERVER')
    port = os.getenv('REMOTE_PORT')
    user = os.getenv('REMOTE_USER')
    remote_path = os.getenv('REMOTE_PATH')

    # -a 递归传输文件，带信息
    # -v 显示传输过程
    # --delete 删除接收端没有的文件
    # --exclude 排除文件
    # -e 指定ssh端口
    # -n 模拟传输过程
    execute_command(f"rsync -av --delete -e 'ssh -p {port}' {args.out} {user}@{remote_server}:{remote_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=str, default="train", choices=["train", "eval", "export", "plot", "rsync"], help="start phase: train, eval, export, plot, rsync")
    parser.add_argument("--fast", action="store_true", help="use fast mode for eval (stage 1 -> stage 2)")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2], help="stage_0: eval all; stage_one: only eval default model in adaptive_action and traitor; stage_two:load adv model to eval.")
    parser.add_argument('--rsync', action='store_true', help='Whether to rsync the output dir to remote server')
    parser.add_argument('--end', action='store_true', help='Run continuously from the current execution until end.')
    args, unknown_args = parser.parse_known_args()
    # 将未知参数转换为字符串
    extra = " ".join(unknown_args)
    args.extra = extra

    if args.phase == "train":
        # train
        train(args)

        if args.end:
            # eval
            if args.fast:
                # eval stage 1
                eval_stage_1(args)
                # eval stage 2
                eval_stage_2(args)

            else:
                eval_all(args)

            # export
            export(args)
            
            # plot
            plot(args)
    
    elif args.phase == "eval":
        if args.stage == 1 or args.fast:
            eval_stage_1(args)
            eval_stage_2(args)

        elif args.stage == 2:
            eval_stage_2(args)

        else:
            eval_all(args)
        
        if args.end:
            # export
            export(args)

            # plot
            plot(args)
    
    elif args.phase == "export":
        # export
        export(args)

        if args.end:
            # plot
            plot(args)

    elif args.phase == "plot":
        # plot
        plot(args)

    elif args.phase == "rsync":
        # rsync
        rsync(args)

    # sync to remote when --rsync and not in rsync phase
    if args.rsync and args.phase != "rsync":
        rsync(args)