

import argparse
import os
import subprocess


def execute_command(command):
    print(f"\033[32m==> {command}\033[0m")
    subprocess.run(command, shell=True)


def train(args, extra, cfgstr):
    execute_command(f"python generate.py train -e {args.env} -s {args.scenario} -a {args.algo} -o {args.out} {extra} {cfgstr}")
    execute_command(f"python parallel.py -s {args.out}/scripts/train_{args.env}_{args.scenario}_{args.algo}.sh -n {args.num_workers}")


def eval_all(args, extra):
    execute_command(f"python generate.py eval -e {args.env} -s {args.scenario} -a {args.algo} -o {args.out} {extra}")
    execute_command(f"python parallel.py -s {args.out}/scripts/eval_{args.env}_{args.scenario}_{args.algo}.sh -n {args.num_workers}")


def eval_stage_1(args, extra):
    execute_command(f"python generate.py eval -e {args.env} -s {args.scenario} -a {args.algo} -o {args.out} {extra} --stage 1")
    execute_command(f"python parallel.py -s {args.out}/scripts/eval_{args.env}_{args.scenario}_{args.algo}_stage_1.sh -n {args.num_workers}")


def eval_stage_2(args, extra):
    execute_command(f"python generate.py eval -e {args.env} -s {args.scenario} -a {args.algo} -o {args.out} {extra} --stage 2")
    execute_command(f"python parallel.py -s {args.out}/scripts/eval_{args.env}_{args.scenario}_{args.algo}_stage_2.sh -n {args.num_workers}")


def export(args):
    execute_command(f"python export.py -e {args.env} -s {args.scenario} -a {args.algo} -o {args.out}")


def plot(args):
    execute_command(f"python plot.py -e {args.env} -s {args.scenario} -a {args.algo} -o {args.out} {'--rsync' if args.rsync else ''}")


def get_paths(args):
    # 往下walk三级目录，返回一个(env, scenario, algo)三元组的列表
    envs = []
    for env_name in os.listdir(os.path.join(args.out, "data")):
        if not os.path.isdir(os.path.join(args.out, "data", env_name)):
            continue
        if args.env is not None and args.env != env_name:
            continue

        for scenario_name in os.listdir(os.path.join(args.out, "data", env_name)):
            if not os.path.isdir(os.path.join(args.out, "data", env_name, scenario_name)):
                continue
            if args.scenario is not None and args.scenario != scenario_name:
                continue

            for algo_name in os.listdir(os.path.join(args.out, "data", env_name, scenario_name)):
                if not os.path.isdir(os.path.join(args.out, "data", env_name, scenario_name, algo_name)):
                    continue
                if args.algo is not None and args.algo != algo_name:
                    continue

                envs.append((env_name, scenario_name, algo_name))
    return envs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default=None, help="env name")
    parser.add_argument("-s", "--scenario", type=str, default=None, help="scenario or map name")
    parser.add_argument("-a", "--algo", type=str, default=None, help="algo name")
    parser.add_argument("-o", "--out", type=str, default="out", help="out dir")
    parser.add_argument("-n", "--num_workers", default=2, type=int, help="Number of workers to use for parallel execution.")
    parser.add_argument("-p", "--phase", type=str, default="train", choices=["train", "eval", "export", "plot"], help="start phase: train, eval, export, plot")
    parser.add_argument("--fast", action="store_true", help="use fast mode for eval (stage 1 -> stage 2)")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2], help="stage_0: eval all; stage_one: only eval default model in adaptive_action and traitor; stage_two:load adv model to eval.")
    parser.add_argument("-c", "--config_path", type=str, default=None, help="default config path")
    parser.add_argument("-f", "--trick", type=str, default=None, help="only generate the specified trick scripts")
    parser.add_argument("-m", "--method", type=str, default=None, help="only generate the specified attack algo scripts")
    parser.add_argument('--rsync', action='store_true', help='Whether to rsync the output dir to remote server')
    args, _ = parser.parse_known_args()

    # 校验
    if args.phase not in ("export", "plot"):
        if args.env is None:
            raise ValueError("env cannot be None")
        if args.scenario is None:
            raise ValueError("scenario cannot be None")
        if args.algo is None:
            raise ValueError("algo cannot be None")

    extra = ""
    if args.trick is not None:
        extra += " --trick {}".format(args.trick)
    if args.method is not None:
        extra += " --method {}".format(args.method)
    
    cfgstr = ""
    if args.config_path is not None:
        cfgstr += " --config_path {}".format(args.config_path)

    if args.phase == "train":
        # train
        train(args, extra, cfgstr)
        # eval
        if args.fast:
            # eval stage 1
            eval_stage_1(args, extra)
            # eval stage 2
            eval_stage_2(args, extra)

        else:
            eval_all(args, extra)

        # export
        export(args)
        
        # plot
        plot(args)
    
    elif args.phase == "eval":
        if args.stage == 1 or args.fast:
            eval_stage_1(args, extra)
            eval_stage_2(args, extra)

        elif args.stage == 2:
            eval_stage_2(args, extra)

        else:
            eval_all(args, extra)
        
        # export
        export(args)

        # plot
        plot(args)
    
    elif args.phase == "export":
        
        for env_name, scenario_name, algo_name in get_paths(args):
            args.env = env_name
            args.scenario = scenario_name
            args.algo = algo_name

            # export
            export(args)
            # plot
            plot(args)

    elif args.phase == "plot":

        for env_name, scenario_name, algo_name in get_paths(args):
            args.env = env_name
            args.scenario = scenario_name
            args.algo = algo_name
            
            # plot
            plot(args)
    
    # rsync
    if args.rsync:
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