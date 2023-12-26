

import argparse
import subprocess

def execute_command(command):
    print("==>Executing...: ", command)
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="smac", help="env name")
    parser.add_argument("-s", "--scenario", type=str, default="3m", help="scenario or map name")
    parser.add_argument("-a", "--algo", type=str, default="mappo", help="algo name")
    parser.add_argument("-n", "--num_workers", default=2, type=int, help="Number of workers to use for parallel execution.")
    parser.add_argument("-p", "--phase", type=str, default="train", choices=["train", "eval", "export", "plot"], help="start phase: train, eval, export, plot")
    parser.add_argument("--fast", action="store_true", help="use fast mode for eval (stage 1 -> stage 2)")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2], help="stage_0: eval all; stage_one: only eval default model in adaptive_action and traitor; stage_two:load adv model to eval.")
    parser.add_argument("-c", "--config_path", type=str, default=None, help="default config path")
    parser.add_argument("-f", "--trick", type=str, default=None, help="only generate the specified trick scripts")
    parser.add_argument("-m", "--method", type=str, default=None, help="only generate the specified attack algo scripts")
    args = parser.parse_args()
    argv = vars(args)

    argstr = ""
    if args.trick is not None:
        argstr += " --trick {}".format(args.trick)
    if args.method is not None:
        argstr += " --method {}".format(args.method)
    
    cfgstr = ""
    if args.config_path is not None:
        cfgstr += " --config_path {}".format(args.config_path)

    if args.phase == "train":
        # train
        execute_command(f"python generate.py train -e {args.env} -s {args.scenario} -a {args.algo} -o ./tmp {argstr} {cfgstr}")
        execute_command(f"cat ./tmp/train_{args.env}_{args.scenario}_{args.algo}.sh | parallel -j {args.num_workers} 2>> errors.txt")
        
        # eval
        if args.fast:
            # eval stage 1
            execute_command(f"python generate.py eval -e {args.env} -s {args.scenario} -a {args.algo} -o ./tmp {argstr} --stage 1")
            execute_command(f"cat ./tmp/eval_{args.env}_{args.scenario}_{args.algo}_stage_1.sh | parallel -j {args.num_workers} 2>> errors.txt")
            # eval stage 2
            execute_command(f"python generate.py eval -e {args.env} -s {args.scenario} -a {args.algo} -o ./tmp {argstr}  --stage 2")
            execute_command(f"cat ./tmp/eval_{args.env}_{args.scenario}_{args.algo}_stage_2.sh | parallel -j {args.num_workers} 2>> errors.txt")
        else:
            execute_command(f"python generate.py eval -e {args.env} -s {args.scenario} -a {args.algo} -o ./tmp {argstr}")
        
        # export
        execute_command(f"python export.py -e {args.env} -s {args.scenario} -a {args.algo}")  

        # plot
        execute_command(f"python plot.py -e {args.env} -s {args.scenario} -a {args.algo}")
    
    elif args.phase == "eval":
        # eval
        if args.fast:
            # eval stage 1
            execute_command(f"python generate.py eval -e {args.env} -s {args.scenario} -a {args.algo} -o ./tmp {argstr} --stage 1")
            execute_command(f"cat ./tmp/eval_{args.env}_{args.scenario}_{args.algo}_stage_1.sh | parallel -j {args.num_workers} 2>> errors.txt")
            # eval stage 2
            execute_command(f"python generate.py eval -e {args.env} -s {args.scenario} -a {args.algo} -o ./tmp {argstr}  --stage 2")
            execute_command(f"cat ./tmp/eval_{args.env}_{args.scenario}_{args.algo}_stage_2.sh | parallel -j {args.num_workers} 2>> errors.txt")
        else:
            execute_command(f"python generate.py eval -e {args.env} -s {args.scenario} -a {args.algo} -o ./tmp {argstr}")
        
        # export
        execute_command(f"python export.py -e {args.env} -s {args.scenario} -a {args.algo}")  

        # plot
        execute_command(f"python plot.py -e {args.env} -s {args.scenario} -a {args.algo}")
    
    elif args.phase == "export":
        # export
        execute_command(f"python export.py -e {args.env} -s {args.scenario} -a {args.algo}")  

        # plot
        execute_command(f"python plot.py -e {args.env} -s {args.scenario} -a {args.algo}")

    elif args.phase == "plot":
        # plot
        execute_command(f"python plot.py -e {args.env} -s {args.scenario} -a {args.algo}")
    