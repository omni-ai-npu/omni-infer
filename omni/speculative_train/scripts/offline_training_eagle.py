import argparse
import hashlib
import math
import os
import time
from collections import defaultdict

import torch
import torch_npu


import torch.distributed as dist
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from tqdm import tqdm
from transformers import AutoTokenizer


from omni.speculative_train.models.auto import AutoDraftModelConfig, AutoEagleDraftModel

from omni.speculative_train.specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from omni.speculative_train.specforge.utils import (
    create_draft_config_from_target,
    get_last_checkpoint,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
)

from omni.speculative_train.specforge.modeling.target.target_head import TargetHead
from omni.speculative_train.specforge.optimizer import BF16Optimizer
from omni.speculative_train.specforge.tracker import create_tracker, get_tracker_class

from omni.speculative_train.data.dataset import (
    build_offline_eagle_dataset,
    prepare_dp_dataloaders,
)
from omni.speculative_train.models.shell import OfflineEagleModel



from vllm.config import ParallelConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle with offline data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument(
        "--draft-model-config",
        type=str,
        required=False,
        help="Draft model config path. If not provided, will auto-generate from target model.",
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )
    parser.add_argument(
        "--lm-head-key",
        type=str,
        default="lm_head.weight",
        help="The key of the lm head weight to load from the target model",
    )

    # add training-related arguments
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--draft-global-batch-size", type=int, default=16)
    parser.add_argument("--draft-micro-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.015)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total training steps. If not provided, will be calculated as num_epochs * steps_per_epoch",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--log-steps", type=int, default=50, help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )
    parser.add_argument("--use-topk-loss", action="store_true")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--decay-factor", type=float, default=0.8)
    parser.add_argument("--draft-attention-backend", type=str, default="flex_attention")
    # data processing type
    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
    )

    # distributed training
    parser.add_argument("--tp-size", type=int, default=1)

    # other args
    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )
    # resume
    parser.add_argument("--resume", action="store_true")

    # report backend
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        choices=["wandb", "tensorboard", "swanlab", "mlflow", "none"],
        help="The integration to report results and logs to.",
    )
    # wandb-specific args
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="The project name for W&B."
    )
    parser.add_argument(
        "--wandb-name", type=str, default=None, help="The run name for W&B."
    )
    parser.add_argument("--wandb-key", type=str, default=None, help="W&B API key.")
    # add swanlab-specific args ---
    parser.add_argument(
        "--swanlab-project",
        type=str,
        default=None,
        help="The project name for swanlab.",
    )
    parser.add_argument(
        "--swanlab-name",
        type=str,
        default=None,
        help="The experiment name for swanlab.",
    )
    parser.add_argument(
        "--swanlab-key",
        type=str,
        default=None,
        help="The API key for swanlab non-interactive login.",
    )
    # mlflow-specific args
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="The MLflow tracking URI. If not set, uses MLFLOW_TRACKING_URI environment variable or defaults to local './mlruns'.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="The MLflow experiment name. If not set, uses MLFLOW_EXPERIMENT_NAME environment variable.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="The MLflow run name. If not set, MLflow will auto-generate one.",
    )

    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-start-step", type=int, default=30)
    parser.add_argument("--profile-num-steps", type=int, default=4)
    parser.add_argument("--profile-record-shapes", action="store_true")
    parser.add_argument("--profile-dir", type=str, default="./trace")

    args = parser.parse_args()

    return parser, args

def main():
    # initialize
    parser, args = parse_args()
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    args.dp_size = dist.get_world_size() // args.tp_size

    args.draft_accumulation_steps = (
        args.draft_global_batch_size // args.dp_size // args.draft_micro_batch_size
    )
    assert (
        args.draft_accumulation_steps * args.draft_micro_batch_size * args.dp_size
        == args.draft_global_batch_size
    ), f"draft_global_batch_size={args.draft_global_batch_size} must be divisible by dp_size={args.dp_size} and micro_batch_size={args.draft_micro_batch_size}"
    print_with_rank(
        f"draft_accumulation_steps={args.draft_global_batch_size} // {args.dp_size} // {args.draft_micro_batch_size}={args.draft_accumulation_steps}"
    )

    # Validate report backend arguments
    tracker_class = get_tracker_class(args.report_to)
    if tracker_class:
        tracker_class.validate_args(parser, args)
    else:
        parser.error(f"Unknown tracker: {args.report_to}")

    tracker = create_tracker(args, args.output_dir)

    # detecting last ckpt for draft model
    draft_model_last_checkpoint = None
    if args.checkpoint is not None:
        draft_model_last_checkpoint = args.checkpoint
        print_on_rank0(f"Drafter will be trained start from {draft_model_last_checkpoint}")
    elif args.resume and os.path.isdir(args.output_dir):
        print_on_rank0(args.output_dir)
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    # build target and draft model
    target_head = TargetHead(args.target_model_path)
    target_head.load_weights(
        model_path=args.target_model_path,
        lm_head_key=args.lm_head_key,
        cache_dir=args.cache_dir,
    )
    target_head.freeze_weights()
    target_head = target_head.eval().npu().to(torch.bfloat16)
    print_with_rank("Initialized target head")

    if draft_model_last_checkpoint is None:
        config = AutoDraftModelConfig.from_file(args.draft_model_config)
        draft_model = AutoEagleDraftModel.from_config(
            config,
        ).npu().to(torch.bfloat16)
    else:
        draft_model = AutoEagleDraftModel.from_pretrained(
            draft_model_last_checkpoint,
        ).npu().to(torch.bfloat16)

    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    print_with_rank("Initialized draft model")

    with rank_0_priority():
        train_eagle3_dataset = build_offline_eagle_dataset(
            args.train_data_path,
            args.max_length,
            'pt',
        )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.draft_micro_batch_size,
        num_workers=4,
        shuffle=True,
        process_group=get_dp_group(),
        pin_memory=True,
    )
    print_with_rank("Initialized train dataloader")

    # Calculate total steps if not provided
    if args.total_steps is None:
        steps_per_epoch = math.ceil(
            len(train_dataloader) / args.draft_accumulation_steps
        )
        args.total_steps = args.num_epochs * steps_per_epoch
        print_with_rank(
            f"Auto-calculated total_steps: {args.total_steps} (num_epochs={args.num_epochs} * steps_per_epoch={steps_per_epoch})"
        )
    else:
        print_with_rank(f"Using provided total_steps: {args.total_steps}")

    # build Eagle3 model
    eagle_model = OfflineEagleModel(
        target_head=target_head,
        draft_model=draft_model,
        length=args.ttt_length,
        attention_backend=args.draft_attention_backend,
        topk=args.topk,
    )
    eagle_model = FSDP(
        eagle_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            keep_low_precision_grads=False,
        ),
        sharding_strategy=ShardingStrategy.NO_SHARD,
        ignored_modules=[],
        process_group=get_dp_group(),
    )
    print_with_rank(f"Initialized Eagle FSDP model")
    global_step, batch_index = 0, 0
    log_dict = defaultdict(float)
    # build other components
    optimizer = BF16Optimizer(
        eagle_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
    )
    print_with_rank("Initialized optimizer and scheduler")

    last_time = time.time()

    # start running
    for epoch in range(args.num_epochs):
        # Run training
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()

        epoch_acces = [[] for _ in range(args.ttt_length)]
        epoch_plosses = [[] for _ in range(args.ttt_length)]
        epoch_topk_losses = [[] for _ in range(args.ttt_length)]

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for data in progress_bar:
            batch_index += 1
            if args.profile:
                if batch_index == args.profile_start_step:
                    print("Start profile")
                    output_dir = os.path.join(
                        args.profile_dir,
                        f"debug_rank{torch.distributed.get_rank()}_{time.time()}/",
                    )
                    experimental_config = torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                    )
                    torch_profiler = torch_npu.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU,
                        ],
                        with_stack=True,
                        record_shapes=args.profile_record_shapes,
                        profile_memory=True,
                        experimental_config=experimental_config,
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir),
                    )
                    torch_profiler.start()
                if batch_index == args.profile_start_step + args.profile_num_steps:
                    output_path = os.path.join(
                        args.profile_dir,
                        f"debug_rank{torch.distributed.get_rank()}_{time.time()}.trace.json",
                    )
                    print(f"End profile {output_path=}")
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(output_path)

            plosses, acces, topk_losses = eagle_model(
                input_ids=data["input_ids"].npu(),  # [B, S]
                attention_mask=data["attention_mask"].npu(),  # [B, S]
                loss_mask=data["loss_mask"].unsqueeze(-1).npu(),  # [B, S, 1] This is different from the online version
                hidden_states=data["hidden_states"].npu(),  # [B, S, D]
            )
            acces = torch.stack(acces).cpu().tolist()

            # calculate weighted loss
            
            if args.use_topk_loss:
                loss_weight = [args.decay_factor**i for i in range(len(topk_losses))]
                loss = (
                    sum([loss_weight[i] * topk_losses[i] for i in range(len(topk_losses))])
                    / args.draft_accumulation_steps
                )
            else:
                loss_weight = [args.decay_factor**i for i in range(len(plosses))]
                loss = (
                    sum([loss_weight[i] * plosses[i] for i in range(len(plosses))])
                    / args.draft_accumulation_steps
                )
            loss.backward()
            log_dict["train/lr"] = optimizer.get_learning_rate()
            for i in range(len(plosses)):
                log_dict[f"train/ploss_{i}"] += (
                    plosses[i].item() / args.draft_accumulation_steps
                )
            for i in range(len(acces)):
                log_dict[f"train/acc_{i}"] += acces[i] / args.draft_accumulation_steps
            for i in range(len(topk_losses)):
                log_dict[f"train/ploss_{i}"] += (
                    topk_losses[i].item() / args.draft_accumulation_steps
                )
            if batch_index % args.draft_accumulation_steps == 0:
                optimizer.step()
                global_step += 1
                # Pass global_step to the tracker
                if global_step % args.log_steps == 0:
                    tracker.log(log_dict, step=global_step)
                log_dict = defaultdict(float)

            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [
                epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
            ]
            epoch_topk_losses = [
                epoch_topk_losses[i] + [topk_losses[i].item()] for i in range(len(topk_losses))
            ]

            if dist.get_rank() == 0:
                avg_loss = sum(pl.item() for pl in plosses) / len(plosses)
                avg_topk_loss = sum(pl.item() for pl in topk_losses) / len(topk_losses)
                avg_acc = sum(acces) / len(acces)
                progress_bar.set_postfix(
                    {"loss": f"{avg_loss:.2f}`",
                     "topk_loss": f"{avg_topk_loss:.2f}",
                     "acc": f"{avg_acc:.2f}"}
                )

        # Log epoch-level training metrics
        train_epoch_logdict = {}
        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).npu().mean()
            dist.all_reduce(acc_i)
            acc_i = (acc_i / dist.get_world_size()).item()
            train_epoch_logdict[f"train/epoch_acc_{i}"] = acc_i
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
            )
        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).npu().mean()
            dist.all_reduce(loss_i)
            loss_i = (loss_i / dist.get_world_size()).item()
            train_epoch_logdict[f"train/epoch_ploss_{i}"] = loss_i
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
            )
        for i in range(len(epoch_topk_losses)):
            loss_i = torch.tensor(epoch_topk_losses[i]).npu().mean()
            dist.all_reduce(loss_i)
            loss_i = (loss_i / dist.get_world_size()).item()
            train_epoch_logdict[f"train/epoch_ploss_{i}"] = loss_i
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i}, topkpLoss: {loss_i:.2f}"
            )
        tracker.log(train_epoch_logdict, step=global_step)

        if epoch % args.save_interval == 0:
            # Save the model
            epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")

            if dist.get_rank() == 0:
                os.makedirs(epoch_output_dir, exist_ok=True)
            dist.barrier()

            with FSDP.state_dict_type(eagle_model, StateDictType.FULL_STATE_DICT):
                state_to_save = {
                    "epoch": epoch,
                    "args": args,
                }
                state_to_save.update(optimizer.state_dict())
                draft_model_state_dict = {
                    k.replace("draft_model.", ""): v
                    for k, v in eagle_model.state_dict().items()
                    if "draft_model." in k and "embed" not in k.lower()
                }

                if dist.get_rank() == 0:
                    torch.save(
                        state_to_save,
                        os.path.join(epoch_output_dir, "training_state.pt"),
                    )
                    print_on_rank0(
                        f"Saved full training state to {epoch_output_dir}/training_state.pt"
                    )
                    draft_model.save_pretrained(
                        epoch_output_dir,
                        state_dict=draft_model_state_dict,
                    )
                    print_on_rank0(f"Saved model configuration to {epoch_output_dir}")
                dist.barrier()
                del state_to_save
                del draft_model_state_dict

    # Close the tracker at the end of training
    tracker.close()
    destroy_distributed()

if __name__ == "__main__":
    main()
