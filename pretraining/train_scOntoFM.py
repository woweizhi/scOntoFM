#!/usr/bin/env python
"""scOntoFM pretraining: MLM + dynamic-margin triplet loss + ontology alignment.

Launch with DeepSpeed (single node, the released model used 6 GPUs):

    deepspeed --num_gpus=6 --master_port 29600 train_scOntoFM.py \
        --dataset_path /path/to/scLLM_onto_dataset \
        --triplet_path ../assets/triplet_ind_hard_top50.pkl

Each optimization micro-step takes a batch of (anchor, positive, negative) cells:
  * MLM loss on the (masked) anchor;
  * dynamic-margin triplet loss on <cls> embeddings of anchor / positive / negative;
  * ontology alignment loss (soft-Spearman between in-batch embedding distances and
    LCA ontology distances) on the anchor <cls> embeddings.

Outputs: ``<output_dir>/checkpoints_<experiment_name>/checkpoint-<step>`` (every 1/4 epoch)
and ``<output_dir>/final_model_<experiment_name>`` (HF ``BertForMaskedLM`` format).
"""

import argparse
import datetime
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import deepspeed
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertConfig, BertForMaskedLM

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from data import TripletCollator, TripletDataset  # noqa: E402
from losses import compute_ontology_alignment_loss, dynamic_triplet_loss, get_cls_embedding  # noqa: E402
from ontology_targets import load_ontology_targets  # noqa: E402

os.environ.setdefault("NCCL_TIMEOUT", "3600")

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
logger = logging.getLogger("scOntoFM.pretrain")


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="scOntoFM pretraining")
    g = p.add_argument_group("data")
    g.add_argument("--dataset_path", required=True,
                   help="Tokenized pretraining corpus (HF dataset with input_ids, cell_type_ontology_term_id).")
    g.add_argument("--triplet_path", default=str(REPO_ROOT / "assets" / "triplet_ind_hard_top50.pkl"))
    g.add_argument("--token_dict_path", default=str(REPO_ROOT / "resources" / "vocab" / "token_dictionary.pkl"))
    g.add_argument("--lca_path", default=str(REPO_ROOT / "assets" / "celltype_similarity_LCA_deepest_allnodes.pkl"))
    g.add_argument("--obo_path", default=str(REPO_ROOT / "resources" / "ontology" / "cl-basic.obo"))

    g = p.add_argument_group("model")
    g.add_argument("--hidden_size", type=int, default=256)
    g.add_argument("--num_hidden_layers", type=int, default=6)
    g.add_argument("--num_attention_heads", type=int, default=4)
    g.add_argument("--intermediate_size", type=int, default=512)
    g.add_argument("--dropout", type=float, default=0.02)
    g.add_argument("--max_position_embeddings", type=int, default=2002)

    g = p.add_argument_group("optimization")
    g.add_argument("--batch_size", type=int, default=64, help="Micro-batch size per GPU.")
    g.add_argument("--epochs", type=int, default=3)
    g.add_argument("--grad_accum", type=int, default=4)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight_decay", type=float, default=1e-3)
    g.add_argument("--warmup_ratio", type=float, default=0.01)
    g.add_argument("--triplet_alpha", type=float, default=1.0, help="Weight of the triplet loss.")
    g.add_argument("--ontoAlign_beta", type=float, default=1.0, help="Weight of the ontology-alignment loss.")
    g.add_argument("--margin_alpha", type=float, default=1.0, help="Scale of the dynamic triplet margin.")
    g.add_argument("--num_workers", type=int, default=8)
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("logging / output")
    g.add_argument("--experiment_name", default="centroid_spearman_dynamicMargin_top50")
    g.add_argument("--output_dir", default="./results")
    g.add_argument("--log_dir", default="./logs")
    g.add_argument("--wandb_project", default="scOntoFM")
    g.add_argument("--no_wandb", action="store_true")
    g.add_argument("--log_every", type=int, default=500)
    g.add_argument("--local_rank", type=int, default=local_rank, help="Set by the DeepSpeed launcher.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #
def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(log_dir, f"pretrain_rank_{local_rank}.log")),
                  logging.StreamHandler()],
    )
    if local_rank == 0:
        logger.info(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | "
                    f"GPUs {torch.cuda.device_count()} | world size {world_size}")


def setup_distributed():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://",
                                             world_size=world_size, rank=local_rank)
    torch.cuda.synchronize()
    if local_rank == 0:
        print(f"Initialized process group: world_size={world_size}")
    torch.distributed.barrier()


def cleanup():
    try:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.cuda.synchronize()
            torch.distributed.destroy_process_group()
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[Rank {local_rank}] Cleanup error: {e}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #
def build_model(args, token_dict) -> BertForMaskedLM:
    config = BertConfig(
        vocab_size=len(token_dict),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_act="relu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_position_embeddings,
        pad_token_id=token_dict["<pad>"],
        output_hidden_states=True,
        return_dict=True,
    )
    return BertForMaskedLM(config)


def build_ds_config(args, num_gpus: int, total_steps: int, warmup_steps: int) -> dict:
    return {
        "train_batch_size": args.batch_size * num_gpus * args.grad_accum,
        "gradient_accumulation_steps": args.grad_accum,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": args.lr, "weight_decay": args.weight_decay, "betas": (0.9, 0.999), "eps": 1e-8},
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {"total_num_steps": total_steps, "warmup_min_lr": 0,
                       "warmup_max_lr": args.lr, "warmup_num_steps": warmup_steps},
        },
        "fp16": {"enabled": True, "initial_scale_power": 12, "loss_scale_window": 1000,
                 "hysteresis": 2, "min_loss_scale": 1},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 5e6,
            "reduce_bucket_size": 5e6,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "round_robin_gradients": True,
        },
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "zero_force_ds_cpu_optimizer": True,
        "memory_efficient_linear": True,
        "wall_clock_breakdown": False,
        "memory_breakdown": False,
        "steps_per_print": 100,
        "verbose": False,
        "logging": {"detailed_logging": False},
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main(args):
    setup_distributed()
    set_seed(args.seed)

    use_wandb = (local_rank == 0) and not args.no_wandb
    if use_wandb:
        import wandb
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.experiment_name, config=vars(args))

    # ontology targets
    dis_mat, nodes, cl2idx, depths_list = load_ontology_targets(args.lca_path, args.obo_path)

    with open(args.token_dict_path, "rb") as f:
        token_dict = pickle.load(f)
    filtered_dataset = Dataset.load_from_disk(args.dataset_path)
    with open(args.triplet_path, "rb") as f:
        triplet_indices = pickle.load(f)

    model = build_model(args, token_dict)

    # NOTE: the LR schedule length is computed from the number of cells in the corpus
    # (as for the released model), not from the number of triplets.
    num_gpus = world_size
    total_steps = int(len(filtered_dataset) * args.epochs / args.batch_size / num_gpus / args.grad_accum)
    warmup_steps = int(args.warmup_ratio * total_steps)
    ds_config = build_ds_config(args, num_gpus, total_steps, warmup_steps)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=ds_config
    )

    triplet_ds = TripletDataset(filtered_dataset, triplet_indices)
    train_sampler = DistributedSampler(
        triplet_ds, num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(), shuffle=True,
    )
    train_loader = DataLoader(
        triplet_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=TripletCollator(token_dict, cl2idx),
        persistent_workers=True,
        prefetch_factor=2,
        timeout=120,
    )

    # ontology matrices on device, normalized to [0, 1]
    dis_mat_norm = dis_mat / np.nanmax(dis_mat)
    onto_distances = torch.tensor(dis_mat_norm, device=model_engine.device, dtype=torch.float32)
    cell_type_depths = torch.tensor(depths_list, device=model_engine.device, dtype=torch.float32)
    if cell_type_depths.max() > 0:
        cell_type_depths = cell_type_depths / cell_type_depths.max()

    n_batches = len(train_loader)
    checkpoint_interval = n_batches // 4
    next_checkpoint_step = checkpoint_interval
    ckpt_root = os.path.join(args.output_dir, f"checkpoints_{args.experiment_name}")

    model_engine.train()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        epoch_losses = {"mlm": [], "triplet": [], "onto_align": []}

        for step, batch in enumerate(train_loader):
            anchor = {k: v.to(model_engine.device) for k, v in batch["anchor"].items()}
            positive = {k: v.to(model_engine.device) for k, v in batch["positive"].items()}
            negative = {k: v.to(model_engine.device) for k, v in batch["negative"].items()}

            outputs = {}
            for name, part in (("anchor", anchor), ("positive", positive), ("negative", negative)):
                outputs[name] = model_engine(input_ids=part["input_ids"], attention_mask=part["attention_mask"],
                                             labels=part["labels"], return_dict=True)

            mlm_loss = outputs["anchor"].loss
            emb_a = get_cls_embedding(outputs["anchor"].hidden_states)
            emb_p = get_cls_embedding(outputs["positive"].hidden_states)
            emb_n = get_cls_embedding(outputs["negative"].hidden_states)

            onto_align_loss = compute_ontology_alignment_loss(emb_a, anchor["cell_type_ids"], onto_distances)
            triplet_loss = dynamic_triplet_loss(emb_a, emb_p, emb_n,
                                                anchor["cell_type_ids"], negative["cell_type_ids"],
                                                onto_distances, cell_type_depths,
                                                margin_alpha=args.margin_alpha)
            total_loss = mlm_loss + args.triplet_alpha * triplet_loss + args.ontoAlign_beta * onto_align_loss

            model_engine.backward(total_loss)
            torch.cuda.synchronize()
            model_engine.step()
            torch.cuda.synchronize()
            if step % 100 == 0:
                torch.cuda.empty_cache()

            epoch_losses["mlm"].append(mlm_loss.item())
            epoch_losses["triplet"].append(triplet_loss.item())
            epoch_losses["onto_align"].append(onto_align_loss.item())

            if local_rank == 0 and step % args.log_every == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}] Step [{step}], Total Loss: {total_loss.item():.4f}")
                if use_wandb:
                    wandb.log({
                        "mlm_loss": mlm_loss.item(),
                        "dynamic_triplet_loss": triplet_loss.item(),
                        "onto_align": onto_align_loss.item(),
                        "total_loss": total_loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    })

            # checkpoint every quarter epoch (counted in micro-steps)
            global_step = epoch * n_batches + step
            if global_step >= next_checkpoint_step:
                torch.distributed.barrier()
                if local_rank == 0:
                    try:
                        os.makedirs(ckpt_root, exist_ok=True)
                        model_engine.module.save_pretrained(os.path.join(ckpt_root, f"checkpoint-{global_step}"))
                        print(f"Saved checkpoint at step {global_step}")
                    except Exception as e:
                        print(f"Error saving checkpoint: {e}")
                next_checkpoint_step = min(next_checkpoint_step + checkpoint_interval, n_batches * args.epochs)
                torch.distributed.barrier()

        if local_rank == 0:
            print(f"Epoch {epoch + 1}: " + ", ".join(
                f"Avg {k} loss = {np.mean(v):.4f}" for k, v in epoch_losses.items()))

    torch.distributed.barrier()
    if local_rank == 0:
        final_dir = os.path.join(args.output_dir, f"final_model_{args.experiment_name}")
        os.makedirs(final_dir, exist_ok=True)
        model_engine.module.save_pretrained(final_dir)
        info = {
            "total_steps": total_steps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "vocab_size": len(token_dict),
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(final_dir, "training_info.pkl"), "wb") as f:
            pickle.dump(info, f)
        print(f"Final model and training info saved to {final_dir}")
    torch.distributed.barrier()


if __name__ == "__main__":
    cli_args = parse_args()
    setup_logging(cli_args.log_dir)
    try:
        main(cli_args)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup()
