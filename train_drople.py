#  Torch imports
import torch

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
import torch.backends.cudnn as cudnn
import deepspeed

cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os

# Local imports
from data.meta_dataset import MetaDataset
from models.common import Classification
from utils.utils import save_args, load_args
from flags import parser, DATA_FOLDER
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW, SGD
from utils.utils import init_distributed_mode, is_main_process
from transformers import get_scheduler
from transformers.integrations import HfDeepSpeedConfig

from models.drople import DroPLe
from torch.optim.lr_scheduler import LambdaLR
import math

from transformers import AutoModelForCausalLM, AutoTokenizer

from decimal import Decimal

import json
import numpy as np
deepspeed.init_distributed()

def get_cosine_schedule_lambda_with_warmup(
        min_lr, base_lr, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(min_lr / base_lr, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return lr_lambda


def get_cosine_schedule_with_warmup(
        optimizer, min_lrs, base_lrs, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5,
        last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = []
    for min_lr, base_lr in zip(min_lrs, base_lrs):
        lr_lambda.append(
            get_cosine_schedule_lambda_with_warmup(min_lr, base_lr, num_warmup_steps, num_training_steps, num_cycles))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def save_checkpoint(model, epoch, path, seed):
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, path)


def build_optimizer_parameters(config, model):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if
                       'pooler' not in n[0] and 'lora' not in n[0] and 'visual_projection' not in n[0]]
    lora_params = [n for n in model.named_parameters() if 'lora' in n[0] or 'visual_projection' in n[0]]
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'pos_embed','relative_position_bias_table']

    weight_decay = getattr(config, "weight_decay", 0.01)
    optimizer_grouped_parameters = [{
        'params': [p for n, p in param_optimizer if p.requires_grad],
        'weight_decay': weight_decay
    }, {
        'params': [p for n, p in lora_params if p.requires_grad],
        'weight_decay': weight_decay,
        'lr': config.lora_lr
    }
    ]

    return optimizer_grouped_parameters


def main():
    # Get arguments and start logging
    # args = parser.parse_args()
    local_parser = deepspeed.add_config_arguments(parser)
    args = local_parser.parse_args()

    device = torch.device(args.device)

    load_args(args.config, args)

    project_name_suffix = "-{class_emb}-{lr:.1E}-{lr_scheuler}-{prompt_type}-{num_prior_tokens}Pr{llm_prompt_depth}x{num_llm_prompts}P{num_text_ctx}T{num_vis_ctx}V-{bias}Init-{num_template}xTemplate-{dist_type}{lambda_dist:.1f}xDist{randaug}-{betas}-WD{wd}".format(
        class_emb='{:d}X{}'.format(args.num_decoder_layers, "decode"),
        lr=Decimal(args.lr),
        lr_scheuler=args.lr_scheduler if args.lr_scheduler else "constant",
        prompt_type=args.prompt_type,
        num_prior_tokens=args.num_prior_tokens,
        llm_prompt_depth=args.llm_prompt_depth,
        num_llm_prompts=args.num_llm_prompts,
        num_text_ctx=args.num_text_ctx,
        num_vis_ctx=args.num_vis_ctx,
        dist_type=args.distillation_type,
        lambda_dist=args.lambda_dist,
        bias='Bias' if args.token_bias else 'No',
        num_template=args.num_text_template,
        randaug='-RandAug' if args.rand_aug else '',
        betas='-'.join([str(b) for b in args.betas]),
        wd=args.weight_decay,
    )

    project_name_suffix = project_name_suffix + '-Skip' if args.decoder_skip_connection else project_name_suffix
    project_name_suffix = project_name_suffix + '-ConcatPrior' if args.concat_fixed_prompts else project_name_suffix

    args.name = args.name + project_name_suffix

    logpath = os.path.join(args.cv_dir, args.name)
    if is_main_process():
        os.makedirs(logpath, exist_ok=True)
        save_args(args, logpath, args.config)

    with open(args.deepspeed_config, 'r') as fp:
        deepspeed_config = json.load(fp)

    patch_aug_config = None
    if args.patch_aug_config:
        with open(args.patch_aug_config, 'r') as f:
            patch_aug_config = json.load(f)

    dschf = HfDeepSpeedConfig(deepspeed_config)

    # init deepspeed

    model = AutoModelForCausalLM.from_pretrained(args.model_base, device_map='cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)

    # Get dataset
    trainset = MetaDataset(
        phase='train',
        dataset=args.dataset,
        num_shots=args.coop_num_shots,
        seed=args.coop_seed,
        num_template=args.num_text_template,
        rand_aug=args.rand_aug,
        patch_aug_config=patch_aug_config,
        use_imbalance=args.use_imbalance,
        imb_ratio=args.imb_ratio,
        return_images = True
    )

    base_testset = MetaDataset(
        phase='val',
        dataset=args.dataset,
        num_shots=args.coop_num_shots,
        seed=args.coop_seed,
    )



    new_testset = MetaDataset(
        phase='test',
        dataset=args.datasets_test_new,
        num_shots=args.coop_num_shots,
        seed=args.coop_seed,
    )

    classnames = {
        'base': trainset.classnames,
        'new': new_testset.classnames,
    }

    custom_template = {
        'base': trainset.template,
        'new': new_testset.template,
    }

    drople_dataset = {
        'source': trainset,
        'target': new_testset,
    }
    model = DroPLe(drople_dataset, classnames, custom_template, args, model, tokenizer)

    start_epoch = 0
    # Load checkpoint
    if args.load is not None:
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['net'], strict=False)
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)

    parameter_group = build_optimizer_parameters(args, model)

    optimizer = AdamW(
        lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay, params=parameter_group
    )

    model_engine = model.to(device)
    sampler = DistributedSampler(trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True,
                                 drop_last=False, seed=3407)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        sampler=sampler)

    if args.freeze_vit:
        min_lrs = [1e-5]
        base_lrs = [args.lr]
    else:
        min_lrs = [1e-5, max(args.lora_lr / args.lr * 1e-5, 5e-6)]
        base_lrs = [args.lr, args.lora_lr]

    if args.lr_scheduler:
        if args.lr_scheduler == 'cosine':
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                min_lrs=min_lrs,
                base_lrs=base_lrs,
                num_warmup_steps=1,
                num_training_steps=max(args.max_epochs, 20) + 1
            )
        else:
            lr_scheduler = get_scheduler(
                name=args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=max(args.max_epochs, 20))
    else:
        lr_scheduler = None

    base_testloader = torch.utils.data.DataLoader(
        base_testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)

    new_testloader = torch.utils.data.DataLoader(
        new_testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)

    evaluator_base = Classification(args, base_testset.idx2label)
    evaluator_new = Classification(args, new_testset.idx2label)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    all_accuracies = []
    best_hm = 0
    best_epoch = -1
    best_model_path = None
    for epoch in tqdm(range(start_epoch, args.max_epochs), desc='Current epoch'):
        trainloader.sampler.set_epoch(epoch)
        if not args.rawclip:
            train(epoch, model_engine, trainloader, lr_scheduler, device, optimizer, args)
        # train(epoch, model_engine, tokenizer, trainloader, args.use_ema, lr_scheduler, optimizer, device)
        if is_main_process():
            if (epoch + 1) % args.eval_val_every == 0:
                with torch.no_grad():  # todo: might not be needed
                    base_acc = \
                    test(epoch, model_engine, base_testloader, evaluator_base, args, logpath, device, subset='Base')[
                        'accuracy']
                    new_acc = \
                    test(epoch, model_engine, new_testloader, evaluator_new, args, logpath, device, subset='New')[
                        'accuracy']
                    hm = 2 * base_acc * new_acc / (base_acc + new_acc)

                    all_accuracies.append({
                        'epoch': epoch,
                        'base_acc': base_acc,
                        'new_acc': new_acc,
                        'hm': hm
                    })

                    print(f'Base: {base_acc}, New: {new_acc}, HM: {hm}\n')

                    # best HM
                    if hm > best_hm:
                        best_hm = hm
                        best_epoch = epoch
                        if is_main_process():
                            current_dir = os.path.dirname(os.path.abspath(__file__))
                            rawclip_dir = "rawclip" if args.rawclip else "drople"
                            ckptpath = os.path.join(current_dir, "results_ckpt", f"{args.dataset}to{args.datasets_test_new}", "checkpoint", rawclip_dir)
                            os.makedirs(ckptpath, exist_ok=True)
                            best_filename = f'ckptfile_{args.coop_seed}.t7'
                            best_model_path = os.path.join(ckptpath, best_filename)
                            if os.path.exists(best_model_path):
                                os.remove(best_model_path)
                            # save_checkpoint(model_engine, epoch, best_model_path, args.coop_seed)
        dist.barrier()

    if is_main_process():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_path = os.path.join(current_dir, "results_ckpt", f"{args.dataset}to{args.datasets_test_new}", "results")
        os.makedirs(results_path, exist_ok=True)
        accuracies_file = os.path.join(results_path, f'accuracies_seed{args.coop_seed}_bestepoch{best_epoch}.json')
        with open(accuracies_file, 'w') as f:
            json.dump(all_accuracies, f, indent=2)

        for acc in all_accuracies:
            print(
                f"epoch{acc['epoch']}-- base_acc: {acc['base_acc']:.2f}%  new_acc: {acc['new_acc']:.2f}%  hm: {acc['hm']:.2f}%")

        avg_base_acc = np.mean([acc['base_acc'] for acc in all_accuracies])
        avg_new_acc = np.mean([acc['new_acc'] for acc in all_accuracies])
        avg_hm = np.mean([acc['hm'] for acc in all_accuracies])
        print(f"\nAverage-- base_acc: {avg_base_acc:.2f}%  new_acc: {avg_new_acc:.2f}%  hm: {avg_hm:.2f}%")
        print(f"Best HM: {best_hm:.2f}% at epoch {best_epoch}")

        summary = {
            "avg_base_acc": avg_base_acc,
            "avg_new_acc": avg_new_acc,
            "avg_hm": avg_hm,
            "best_hm": best_hm,
            "best_epoch": best_epoch
        }
        with open(accuracies_file, 'r+') as f:
            data = json.load(f)
            data.append(summary)
            f.seek(0)
            json.dump(data, f, indent=2)


def train(epoch, model_engine, trainloader, lr_scheduler, device, optimizer, args):
    '''
    Runs training for an epoch
    '''
    model_engine.train()
    train_loss = 0.0

    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc='Training'):
        if epoch == 0 and idx == len(trainloader) // 10:
            lr_scheduler.step()
        data = [d.to(device) if hasattr(d, 'to') else d for d in data]
        data[0] = data[0].bfloat16()
        data[1] = data[1].bfloat16()

        total_loss, targetpred = model_engine(data, current_epoch=epoch)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

    if lr_scheduler is not None:
        lr_scheduler.step()

    train_loss /= len(trainloader)
    print('Epoch: {} | Loss: {:.5f}'.format(epoch, train_loss))

def test(epoch, model, testloader, evaluator, args, logpath, device, subset):
    '''
    Runs testing for an epoch
    '''

    evaluator.reset()
    model.eval()
    model.compute_all_class_embeddings(subset=subset.lower())

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing on {}'.format(subset)):
        data = [d.to(device) for d in data]
        data[0] = data[0].bfloat16()
        data[1] = data[1].bfloat16()

        with torch.inference_mode():

            _, predictions = model(data, current_epoch=epoch, subset=subset.lower())

        predictions = predictions.cpu()
        evaluator.process(predictions, data[-1].cpu())


    print("Done Running Results")
    stats = evaluator.evaluate()

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name

    print(f'Test Epoch: {epoch}')
    print(result)
    return stats


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
