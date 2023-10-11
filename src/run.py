import argparse
import logging
import os
import sys
import json
import torch

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("run")

from imt_environment.environment import Environment
from imt_environment.imt_system import (
    PrefixTransformer,
    DBATransformer,
    Bitiimt,
    LecaImt,
    ChatgptImt,
)
from imt_environment.policy import (
    MtpePolicy,
    Left2RightPolicy,
    RandomPolicy,
    Left2RightInfillingPolicy,
    RandomInfillingPolicy,
    utils,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--src-path", type=str, required=True, help="file path of source language")
    parser.add_argument("--tgt-path", type=str, required=True, help="file path of target language")
    parser.add_argument("--src-lang", type=str, required=True, help="source language")
    parser.add_argument("--tgt-lang", type=str, required=True, help="target language")
    parser.add_argument("--policy-spm-model", type=str, default=None, help="path of spm model used by policy")
    parser.add_argument("--policy-seed", type=int, default=1, help="random seed for policy")
    parser.add_argument("--export-path", type=str, default=None, help="the export path when using real env")

    parser.add_argument("--policy", default=0, type=int, required=True, help="the type of policy")
    parser.add_argument("--imt", default=0, type=int, required=True, help="the type of imt system")
    parser.add_argument("--imt-args", default=None, type=str, help="the path of imt's args")
    parser.add_argument("--checkpoint", default=None, type=str, help="the path of the checkpoint")

    args = parser.parse_args()
    logger.info("Parameters: {}".format(args))

# Initialize IMT system
imt_type = args.imt
if args.imt_args is not None:
    with open(args.imt_args) as iarg:
        imt_args = json.load(iarg)
else:
    imt_args = args
if imt_type == 0:
    imt_system = PrefixTransformer(imt_args)
elif imt_type == 1:
    imt_system = DBATransformer(imt_args)
elif imt_type == 2:
    imt_system = Bitiimt(imt_args)
elif imt_type == 3:
    imt_system = LecaImt(imt_args)
elif imt_type == 4:
    imt_system = ChatgptImt(imt_args)

# Initialize policy tokenizer
if args.policy_spm_model is not None:
    tokenizer = utils.SentencePieceTokenizer(args.policy_spm_model)
else:
    tokenizer = utils.SpaceTokenizer()

# Initialize policy
policy_type = args.policy
if policy_type == 0:
    policy = MtpePolicy(tokenizer)
elif policy_type == 1:
    policy = Left2RightPolicy(tokenizer, n=1)
elif policy_type == 2:
    policy = RandomPolicy(tokenizer, args.policy_seed)
elif policy_type == 3:
    policy = Left2RightInfillingPolicy(tokenizer)
elif policy_type == 4:
    policy = RandomInfillingPolicy(tokenizer, args.policy_seed)
elif policy_type == 5:
    pass # human policy

if args.checkpoint is not None:
    dir = os.path.dirname(args.checkpoint)
    if not os.path.exists(dir):
        os.makedirs(dir)
if args.export_path is not None:
    dir = os.path.dirname(args.export_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def run_interaction():
    success = []
    turns = []
    avg_editing_cost = []
    normalized_editing_cost = []
    response_time = []
    consistency = []
    num = 0
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint)
        num = state_dict["num"]
        success = state_dict["success"]
        turns = state_dict["turns"]
        avg_editing_cost = state_dict["avg_editing_cost"]
        normalized_editing_cost = state_dict["normalized_editing_cost"]
        response_time = state_dict["response_time"]
        consistency = state_dict["consistency"]

    with open(args.src_path, "r") as src, open(args.tgt_path, "r") as tgt:
        for i, (src_sentence, tgt_sentence) in enumerate(zip(src, tgt)):
            if i < num:
                continue
            logger.info("test case {}".format(i))
            env.initialize_episode(src_sentence, tgt_sentence)
            episode_over = False
            
            while not episode_over:
                episode_over, state = env.next_turn()
            success.append(int(state["success"]))
            turns.append(state["turn"])
            avg_editing_cost.append(state["editing_cost"])
            normalized_editing_cost.append(state["normalized_editing_cost"])
            response_time.append(state["response_time"] / state["turn"])
            consistency.append(state["consistency"])
            if args.checkpoint is not None and (i + 1) % 2 == 0:
                torch.save({
                    "num": i + 1,
                    "success": success,
                    "turns": turns,
                    "avg_editing_cost": avg_editing_cost,
                    "normalized_editing_cost": normalized_editing_cost,
                    "response_time": response_time,
                    "consistency": consistency
                }, args.checkpoint)

    num = i + 1
    logger.info("success rate: {:.3f} | avg turns: {:.2f} | avg editing cost: {:.2f} ({:.2%}) | avg responding time: {:.3f} | avg consistency: {:.2f}".format(
        sum(success) / num,
        sum(turns) / num,
        sum(avg_editing_cost) / num,
        sum(normalized_editing_cost) / num,
        sum(response_time) / num,
        sum(consistency) / sum(t - 1 for t in turns) if sum(t - 1 for t in turns) > 0 else 0
    ))

if policy_type != 5:
    # Initialize environment
    env = Environment(imt_system, policy)
    run_interaction()
elif policy_type == 5:
    with open(args.src_path, "r") as src:
        testset = src.readlines()
    testset = [s.strip() for s in testset]
    from imt_environment.real_environment import create_app
    app = create_app(args, imt_system, testset)
    app.run(host="0.0.0.0", port=5000)