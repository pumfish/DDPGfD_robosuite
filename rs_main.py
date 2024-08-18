import os
import sys
import time
import argparse

sys.path.append('/workspace/S/heguanhua2/robot_rl/robosuite_jimu')

from rs_trainer import RLTrainer


def main():
    # set argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', help='Training Configuration', type=str)
    parser.add_argument('-e', '--eval',
                        help='Evaluation mode',
                        action='store_true',
                        default=False)
    parser.add_argument('-c', '--collect',
                        help='Collect Demonstration Data',
                        action='store_true',
                        default=False)
    parser.add_argument('-n', '--num-collect',
                        help='Number of episode for demo collection',
                        type=int, default=100)
    args = parser.parse_args()
    conf_path = args.conf

    trainer = RLTrainer(conf_path, args.eval)
    if args.eval:
        trainer.eval(save_fig=False)
    elif args.collect:
        trainer.collect_demo(args.num_collect)
    else:
        if trainer.conf.pretrain_demo:
            trainer.pretrain()
        trainer.train()


if __name__ == '__main__':
    os.putenv('DISPLAY', ':0')
    main()

