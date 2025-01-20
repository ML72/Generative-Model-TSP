import os
import time
import random
from tqdm import tqdm
import torch
import numpy as np
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.transformations import transform_tensor_batch, fit_unit_batch
from utils.hardness_adaptive import get_hard_samples
from utils.ewc import EWC
from utils.log_utils import log_values
from utils.distributions import link_batch
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(
        model,
        optimizer,
        baseline,
        lr_scheduler,
        epoch,
        generator,
        z_dataset,
        ewc_dataset,
        val_dataset,
        problem,
        tb_logger,
        opts
):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate training data
    training_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution
    )

    if opts.training_distribution != 'unif':
        data_tensor = torch.zeros((opts.epoch_size, opts.graph_size, 2), dtype=torch.float)

        if opts.training_distribution == 'vae_curriculum' and z_dataset == None:
            z_dataset = torch.randn(opts.epoch_size, generator.latent_dim)

        LINK_VALUES = [1, 5, 10, 15]
        for i in range(opts.epoch_size // opts.batch_size):
            if opts.training_distribution == 'clusters':
                link_size = LINK_VALUES[i % len(LINK_VALUES)]
                data_tensor[i*opts.batch_size:(i+1)*opts.batch_size] = torch.tensor(
                    link_batch(opts.batch_size, opts.graph_size, link_size=link_size, noise=0.05)
                )
            elif opts.training_distribution == 'vae':
                with torch.no_grad():
                    sampled_sequences = generator.sample(num_samples=opts.batch_size, seq_length=opts.graph_size, device=opts.device)
                data_tensor[i*opts.batch_size:(i+1)*opts.batch_size] = sampled_sequences.cpu().detach()
            elif opts.training_distribution == 'vae_curriculum':
                with torch.no_grad():
                    sampled_sequences = generator.sample(
                        num_samples=opts.batch_size,
                        z=z_dataset[i*opts.batch_size:(i+1)*opts.batch_size],
                        seq_length=opts.graph_size,
                        device=opts.device
                    )
                data_tensor[i*opts.batch_size:(i+1)*opts.batch_size] = sampled_sequences.cpu().detach()
            else:
                raise ValueError("Unknown training distribution")

        data_tensor = torch.clamp(data_tensor, 0, 1)
        for i in range(data_tensor.shape[0]):
            training_dataset[i] = data_tensor[i]
    
    # Randomly make a specified fraction of the data harder if hardness adaptive curriculum is used
    if opts.hardness_adaptive_percent > 0:
        target = (training_dataset.size * opts.hardness_adaptive_percent) // 100
        hard_data = get_hard_samples(model, training_dataset.data[:target], eps=5, device=opts.device, baseline=baseline)
        training_dataset.data[:target] = hard_data
    
    # Initialize EWC if applicable
    ewc = None
    if opts.ewc_lambda > 0:
        if ewc_dataset is None or opts.ewc_from_unif or epoch < opts.ewc_warmup_epochs:
            ewc_dataset = torch.FloatTensor(opts.ewc_fisher_n, opts.graph_size, 2).uniform_(0, 1).to(opts.device)
        if opts.ewc_adaptive:
            ewc_dataset = get_hard_samples(
                model, ewc_dataset, eps=5, device=opts.device, baseline=baseline, get_easy=True
            ).to(opts.device)
        bl_cost = 0
        if baseline is not None and hasattr(baseline, 'model'):
            bl_cost = rollout(baseline.model, ewc_dataset, opts).to(opts.device)
        ewc = EWC(
            model,
            bl_cost,
            ewc_dataset,
            opts
        )
    
    # Shuffle dataset before training!
    random.shuffle(training_dataset.data)
    
    # Wrap dataset in DataLoader
    training_dataset_wrapped = baseline.wrap_dataset(training_dataset)
    training_dataloader = DataLoader(training_dataset_wrapped, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            ewc,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

        if opts.verbose_checkpoints:
            print('Saving datasets...')
            data_filepath = os.path.join(opts.save_dir, 'epoch-{}_data.npy'.format(epoch))
            np.save(data_filepath, np.array(training_dataset.data))

    if not opts.no_tensorboard:
        avg_reward = validate(model, val_dataset, opts)
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()
    
    # Compute importance on instances
    calc_ewc = opts.ewc_lambda > 0
    use_curriculum = opts.training_distribution == 'vae_curriculum'
    if calc_ewc or use_curriculum:
        train_importance = rollout(model, training_dataset, opts)
        sorted_idx = torch.argsort(train_importance)
    
    # Create new z dataset if applicable
    if use_curriculum:
        NUM_FOCUS = opts.epoch_size // 5
        NUM_RESAMPLE = opts.epoch_size - NUM_FOCUS
        z_focus = z_dataset[sorted_idx][NUM_RESAMPLE:]
        mean = z_focus.mean(dim=0)
        std = z_focus.std(dim=0)

        z_focus = torch.normal(mean, std, size=(NUM_FOCUS, generator.latent_dim))
        z_resample = torch.randn(NUM_RESAMPLE, generator.latent_dim)
        z_dataset = torch.cat([z_focus, z_resample], dim=0)

    # Create new EWC dataset if applicable
    if calc_ewc:
        ewc_dataset_new = []
        for i in range(opts.ewc_fisher_n):
            ewc_dataset_new.append(training_dataset[sorted_idx[i]])
        ewc_dataset = torch.stack(ewc_dataset_new, dim=0).to(opts.device)

    return z_dataset, ewc_dataset


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        ewc,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)

    # Apply data transformations if specified
    if opts.data_equivariance:
        x = transform_tensor_batch(x)
    
    # Ensure data is in correct range
    x = fit_unit_batch(x)

    # Move to GPU if available
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    loss = (cost - bl_val) * log_likelihood
    if opts.hardness_adaptive_percent > 0:
        w = ((cost/bl_val) * log_likelihood).detach()
        t = torch.FloatTensor([20-(epoch % 20)]).to(loss.device)
        w = torch.tanh(w)
        w /= t
        w = torch.nn.functional.softmax(w, dim=0)
        reinforce_loss = (w * loss).sum()
    else:
        reinforce_loss = (loss).mean()
    loss = reinforce_loss + bl_loss

    if ewc is not None:
        loss += ewc.penalty(model)

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
