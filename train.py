"""
The debug wrapper script.
"""

import argparse
import os
import sys

_ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('--cuda', '-c', type=str, default='0', help='gpu ids, like: 1,2,3')
_ARG_PARSER.add_argument('--name', '-n', type=str, default='debug', help='save name.')
_ARG_PARSER.add_argument('--debug', '-d', default=False, action="store_true")
_ARG_PARSER.add_argument('--config', type=str, default='crosslingual_dep_parser', help='configuration file name.')


_ARGS = _ARG_PARSER.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = _ARGS.cuda

if _ARGS:
    from allennlp.commands import main

config_file = f"config/{_ARGS.config}.jsonnet"

# Use overrides to train on CPU.
overrides = "{\"trainer.cuda_device\":0}"

serialization_dir = "results/" + _ARGS.name

if _ARGS.debug:
    import shutil
    # Training will fail if the serialization directory already
    # has stuff in it. If you are running the same training loop
    # over and over again for debugging purposes, it will.
    # Hence we wipe it out in advance.
    # BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
    shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
argv = [  # command name, not used by main
    "allennlp",
    "train",
    config_file,
    "-s", serialization_dir,
    "-o", overrides,
]

if not _ARGS.debug:
    argv.append("--file-friendly-logging")

print(" ".join(argv))
sys.argv = argv
main()
