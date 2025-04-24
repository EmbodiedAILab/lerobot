import dataclasses
import enum
import logging
import socket
import tyro
import os
import json

import wandb
import torch
if "ASCEND_HOME_PATH" in os.environ:
    import torch_npu
    logging.info("exists npu, import torch_npu")

from lerobot.common.utils.utils import init_logging
from lerobot.common.utils.random_utils import set_seed
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.common.serving.websocket_policy_server import WebsocketPolicyServer
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy



class PolicyType(enum.Enum):
    """Supported environments."""

    ACT = "act"
    DIFFUSION = "diffusion"
    PI0 = "pi0"

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Checkpoint directory (e.g., "outputs/train/act_move_reel_0322_nodepth/checkpoints/040000/pretrained_mode").
    path: str
    
    # policy type
    type: str
    
@dataclasses.dataclass
class Wandb:
    """WanDB config."""

    enable: bool = False
    
    # drop the first frame in order to reduce cold-start influence
    drop_first_n_frames: int = 1

@dataclasses.dataclass
class Trace:
    """Tracing config."""

    enable: bool = False
    
    # drop the first frame in order to reduce cold-start influence
    drop_first_n_frames: int = 1


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint = dataclasses.field(default_factory=Checkpoint)
    
    wandb : Wandb = dataclasses.field(default_factory=Wandb)
    
    trace : Trace = dataclasses.field(default_factory=Trace)

def main(args: Args) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(1000)
    
    logging.info(args)
    
    # policy has been in device and evaluated
    if args.policy.type == PolicyType.ACT.value:
        policy = ACTPolicy.from_pretrained(args.policy.path)
    elif args.policy.type == PolicyType.DIFFUSION.value:
        policy = DiffusionPolicy.from_pretrained(args.policy.path)
    elif args.policy.type == PolicyType.PI0.value:
        policy = PI0Policy.from_pretrained(args.policy.path)
        
    if args.wandb.enable and args.trace.enable:
        logging.warning("enable pref and profiling at the same time, perf data will be influenced!")
    
    if args.wandb.enable:
        tags = [args.policy.type]
        config = dataclasses.asdict(policy.config)
        if "train_config.json" in os.listdir(args.policy.path):
            filepath = os.path.join(args.policy.path, "train_config.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tags.append(data['dataset']['repo_id'])

        if torch.cuda.is_available():
            config.update({"hardware_platform": torch.cuda.get_device_name(torch.cuda.current_device())})
            tags.append('cuda')
        elif torch_npu.npu.is_available():
            config.update({"hardware_platform": torch_npu.npu.get_device_name(torch_npu.npu.current_device())})
            tags.append('npu')

        wandb.init(project="model-inference-monitoring", 
              config=config,
              tags=tags)
        
        wandb.define_metric("infer_cost_ms", summary="min,max,mean")
        
    
                      
    # Record the policy's behavior.
    # if args.record:
    #     policy = _policy.PolicyRecorder(policy, "policy_records")

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={
            'wandb_enable': args.wandb.enable,
            'drop_first_n_frames': args.wandb.drop_first_n_frames,
            'trace_enable': args.trace.enable
        },
    )
    server.serve_forever()


if __name__ == "__main__":
    init_logging()
    main(tyro.cli(Args))
