# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Finetune utilities."""

from functools import partial
import sys
import torch

from megatron import get_args, get_num_microbatches
from megatron import print_rank_0
from megatron import get_timers
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import ModelType
from megatron.utils import average_losses_across_data_parallel_group
from megatron.utils import calc_params_l2_norm
from megatron.utils import check_adlr_autoresume_termination


def process_batch(batch):
    """Process batch and produce inputs for the model."""
    args = get_args()

    tokens = batch['text'].long().cuda().contiguous()
    types = batch['types'].long().cuda().contiguous()
    labels = batch['label'].long().cuda().contiguous()
    attention_mask = batch['padding_mask'].float().cuda().contiguous()
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, types, labels, attention_mask


def cross_entropy_loss_func(labels, output_tensor):
    logits = output_tensor

    # Cross-entropy loss.
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(logits.contiguous().float(), labels)

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def _cross_entropy_forward_step(batch, model):
    """Simple forward step with cross-entropy loss."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch
    tokens, types, labels, attention_mask = process_batch(batch_)
    timers('batch-generator').stop()

    # Forward model.
    output_tensor = model(tokens, attention_mask, tokentype_ids=types)

    return output_tensor, partial(cross_entropy_loss_func, labels)


def build_data_loader(dataset, micro_batch_size, num_workers, drop_last,
        task_collate_fn=None):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    # Sampler.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank)

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=micro_batch_size,
                                              sampler=sampler,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=True,
                                              collate_fn=task_collate_fn)

    return data_loader


def _build_infinite_size_dataloader(dataloader):
    """Build a looped dataloader with infinite size."""

    iterator = dataloader.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = dataloader.__iter__()


def _build_train_valid_dataloaders(train_dataset, valid_dataset, 
    task_collate_fn=None):
    """Traing and validation dataloaders."""
    args = get_args()

    print_rank_0('building train and validation dataloaders ...')
    # Training dataset.
    train_dataloader = build_data_loader(train_dataset, args.micro_batch_size,
                                         args.num_workers, not args.keep_last,
                                         task_collate_fn)
    # Set the training iterations.
    args.train_iters_per_epoch = len(train_dataloader)
    args.train_iters = args.epochs * args.train_iters_per_epoch
    # Validation dataset. For this dataset, we do not need to set up
    # shuffling so we can just use a simple infinite loop.
    valid_dataloader_ = build_data_loader(valid_dataset, args.micro_batch_size,
                                          args.num_workers, not args.keep_last,
                                          task_collate_fn)
    valid_dataloader = _build_infinite_size_dataloader(valid_dataloader_)

    # Now that we've built the data loaders, set batch_size arguments
    # to the actual batch size the model will see for this dataset.
    # This is necessary so pipeline transfers know what size they are
    # and the LR schedule, which is based on samples seen, gets set
    # correctly.
    args.orig_micro_batch_size = args.micro_batch_size
    args.orig_global_batch_size = args.global_batch_size
    if hasattr(train_dataset, 'sample_multiplier'):
        # If our dataset as a sample_multiplier attribute that means
        # each "sample" from the dataset actually has multiple samples
        # that will collapse into the batch dimension (for example in
        # the RACE dataset that has several options), we need to
        # account for that when setting the micro batch size.
        args.micro_batch_size *= train_dataset.sample_multiplier
        args.global_batch_size *= train_dataset.sample_multiplier

    return train_dataloader, valid_dataloader


