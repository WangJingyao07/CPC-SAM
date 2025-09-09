import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, getStat
from torchvision import transforms
from icecream import ic
import wandb

from prompt import Prompt
from medpy import metric
from cal_dice import dice_score


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight: float = 0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


@torch.no_grad()
def validate(args, model, validloader, multimask_output):
    return_all = args.return_all
    score_dice = []
    model.eval()
    dice_list = None
    dice_dict_list = None
    for _, sampled_batch in enumerate(validloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
        low_res_label_batch = sampled_batch['low_res_label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        low_res_label_batch = low_res_label_batch.cuda()

        outputs = model(image_batch, multimask_output, args.img_size)

        low_res_logits = outputs['low_res_logits']
        # dice_list = None
        # dice_dict_list = None
        if return_all:
            if dice_dict_list is None:
                dice_dict_list = {}
            # dice, dice_list = dice_score(low_res_logits, low_res_label_batch, return_all=return_all)
            dice, dice_list = dice_score(low_res_logits, low_res_label_batch, return_all=return_all)
            for item in dice_list.keys():
                if item not in dice_dict_list.keys():
                    dice_dict_list[item] = [dice_list[item]]
                else:
                    dice_dict_list[item].append(dice_list[item])
            # print(dice_list)
        else:
            dice = dice_score(low_res_logits, low_res_label_batch, return_all=return_all)
        score_dice.append(dice.cpu().numpy())
    model.train()
    if dice_dict_list is not None:
        for item in dice_dict_list.keys():
            dice_dict_list[item] = np.array(dice_dict_list[item]).mean()
    return np.mean(score_dice), dice_dict_list


def trainer(args, model, snapshot_path, multimask_output, low_res, test_num=-2000,
            model_num=random.randint(0, 1000000)):
    if args.dataset == 'kvasir':
        from datasets.dataset_kvasir import Synapse_dataset, RandomGenerator
    elif args.dataset == 'hand':
        from datasets.dataset_hand import Synapse_dataset, RandomGenerator
    elif args.dataset == 'car_m':
        from datasets.dataset_car_multi import Synapse_dataset, RandomGenerator
    elif args.dataset == 'cell':
        from datasets.dataset_cell import Synapse_dataset, RandomGenerator
    elif args.dataset == 'lung':
        from datasets.dataset_lung import Synapse_dataset, RandomGenerator
    elif args.dataset in ['brow', 'eye', 'hair', 'nose', 'mouth', 'celeb']:
        from datasets.dataset_celeb import Synapse_dataset, RandomGenerator
    elif args.dataset in ['car', 'wheel', 'window']:
        from datasets.dataset_car import Synapse_dataset, RandomGenerator
    elif args.dataset == 'teeth':
        from datasets.dataset_teeth import Synapse_dataset, RandomGenerator
    elif args.dataset == 'body':
        from datasets.dataset_body import Synapse_dataset, RandomGenerator
    elif args.dataset == 'data1':
        from datasets.dataset_celebAMMask_HQ import Synapse_dataset, RandomGenerator
    else:
        print("##### Unimplemented dataset #####")
        sys.exit()

    exp_name = f"{args.dataset}-{args.num_data}-{args.exp_type}-lora_mask"
    logger = wandb.init(project='Auto-sam', name=exp_name, resume='allow', anonymous='must', mode=args.wandb_mode)
    logger.config.update(vars(args))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    base_lr = args.base_lr
    num_classes = args.num_classes

    db_train = Synapse_dataset(
        train_dir=args.root_path, num_data=args.num_data, dataset=args.dataset,
        transform=transforms.Compose([RandomGenerator(
            output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])
        ]),
    )
    db_test = Synapse_dataset(
        train_dir=args.root_path, num_data=-1, dataset=args.dataset,
        transform=transforms.Compose([RandomGenerator(
            output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])
        ]),
    )

    num_train = int(len(db_train) * 0.5)
    num_valid = len(db_train) - num_train
    selector = range(len(db_train))
    logging.info("The length of train set is: {}".format(num_train))
    logging.info("The length of valid set is: {}".format(num_valid))

    # print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=args.batch_size, num_workers=32, pin_memory=True,
                             worker_init_fn=worker_init_fn, sampler=selector[:num_train])
    validloader = DataLoader(db_train, batch_size=args.batch_size, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn, sampler=selector[num_train:])
    selector_test = range(len(db_test))
    testloader = DataLoader(db_test, batch_size=1, num_workers=32, pin_memory=True,
                            worker_init_fn=worker_init_fn, sampler=selector_test[test_num:])

    model.train()
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    # optimizer = optim.AdamW(
    #     list(p for n, p in model.named_parameters() if (p.requires_grad and ("no_mask_embed" not in n))),
    #     lr=base_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    optimizer = optim.AdamW(
        list(p for n, p in model.named_parameters() if (p.requires_grad and (("no_mask_embed" not in n)
                                                                             and ("iou_token" not in n)
                                                                             and ("mask_tokens" not in n)))),
        lr=base_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=base_lr, total_steps=max_iterations)

    prompt_module = Prompt(model=model, args=args, max_iterations=max_iterations)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    time_show = time.time()
    best_valid_score = -1
    not_increase_num = 0
    # saved_root = args.output + args.dataset + "/"
    saved_root = "./saved_model/" + args.dataset + "/"

    # test_score, test_score_list = validate(args, model, testloader, multimask_output)

    for epoch_num in range(max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()

            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)

            if "attn" in outputs.keys():
                L1_loss = 0
                # attn = outputs["attn"]
                item = outputs["attn"][0][0]
                L1_loss += (torch.norm(item, p=1) /
                            (item.shape[0] * item.shape[1] *
                             item.shape[2] * item.shape[3]))
                item = outputs["attn"][0][2]
                L1_loss += (torch.norm(item, p=1) /
                            (item.shape[0] * item.shape[1] *
                             item.shape[2] * item.shape[3]))
                item = outputs["attn"][1][0]
                L1_loss += (torch.norm(item, p=1) /
                            (item.shape[0] * item.shape[1] *
                             item.shape[2] * item.shape[3]))
                item = outputs["attn"][1][2]
                L1_loss += (torch.norm(item, p=1) /
                            (item.shape[0] * item.shape[1] *
                             item.shape[2] * item.shape[3]))
                alpha = 0.01
                loss = loss + alpha * L1_loss

            if "prompt" in outputs.keys():
                L1_loss1 = 0
                # attn = outputs["attn"]
                item = outputs["prompt"]
                L1_loss1 += (torch.norm(item, p=1) /
                            (item.shape[0] * item.shape[1] *
                             item.shape[2]))
                beta = 0.01
                loss = loss + beta * L1_loss1

            if "prompt_weight" in outputs.keys():
                L1_loss2 = 0
                # attn = outputs["attn"]
                item = outputs["prompt_weight"]
                L1_loss2 += (torch.norm(item, p=1) /
                            (item.shape[0] * item.shape[1] *
                             item.shape[2]))
                # gama = 0.01
                gama = 0.01
                loss = loss + gama * L1_loss2

            # logger.log({'info/stage1_loss': loss})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # show_num = 1
            # if iter_num % show_num == 0:
            #     logging.info('iteration {}/{} : loss : {:.4f}, loss_ce: {:.4f}, '
            #                  'loss_dice: {:.4f}, time:{:4f}s'.format
            #                  (iter_num, len(trainloader), loss.item(),
            #                   loss_ce.item(), loss_dice.item(), time.time() - time_show))
            #     time_show = time.time()

            # t1 = time.time()
            valid_batch = next(iter(validloader))
            # print("t1=", time.time() - t1)
            for param_group in optimizer.param_groups:
                eta = param_group['lr']
                break
            prompt_module.step(sampled_batch, valid_batch, eta, optimizer, unrolled=args.unrolled, cur_iter=iter_num)

            ##### Adjust Learning Rate #####
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (
                            1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            show_num = 10
            if iter_num % show_num == 0:
                logging.info('iteration {} : loss : {:.4f}, loss_ce: {:.4f}, '
                             'loss_dice: {:.4f}, time:{:4f}s'.format
                             (iter_num, loss.item(),
                              loss_ce.item(), loss_dice.item(), time.time() - time_show))
                if "attn" in outputs.keys():
                    print("loss_l1_attn:{:.4f}".format(L1_loss.item()), end=", ")
                if "prompt" in outputs.keys():
                    print("loss_l1_prompt:{:.4f}".format(L1_loss1.item()), end=", ")
                if "prompt_weight" in outputs.keys():
                    print("loss_l1_prompt_weight:{:.4f}".format(L1_loss2.item()), end=", ")
                if "attn" in outputs.keys() or "prompt" in outputs.keys() or "prompt_weight" in outputs.keys():
                    print("")

                time_show = time.time()

                image = image_batch[0, :, :, :]
                output_masks = outputs['masks']
                labs = label_batch[0, ...].unsqueeze(0)

                ims = {}
                image = (image - image.min()) / (image.max() - image.min())
                image = image.mul(255).permute(1, 2, 0).to('cpu').numpy()
                ims['train/Image'] = wandb.Image(image)

                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)[0, ...]
                output_masks = output_masks.mul(255).to('cpu').numpy()
                ims['train/Prediction'] = wandb.Image(output_masks)

                labs = labs.mul(255).to('cpu').numpy()
                ims['train/GroundTruth'] = wandb.Image(labs)

                logger.log(ims)

        # validate the model at every epoch ending
        valid_score, valid_score_list = validate(args, model, validloader, multimask_output)
        print('Epoch %d : valid score : %f' % (epoch_num + 1, valid_score), end="  ")
        if valid_score_list is not None:
            print("valid_score_list:", valid_score_list)
        # logger.log({'info/valid_score': valid_score})

        # 每隔100个epoch测试一次
        if epoch_num % 100 == 0 and 98 < epoch_num < (args.max_epochs - 5):
        # if epoch_num % 10 == 0 and epoch_num >= 50:
            # 加载 state_dict
            model.load_state_dict(torch.load(saved_root + str(model_num) + '.pth'))
            print("load model:", saved_root + str(model_num) + '.pth')
            test_score, test_score_list = validate(args, model, testloader, multimask_output)
            print('test score : %f' % test_score)
            if test_score_list is not None:
                print("test_score_list:", test_score_list)

        if valid_score > best_valid_score:
            best_valid_score = valid_score
            print("get best model, best_valid_score = ", best_valid_score)
            not_increase_num = 0
            # 保存模型的 state_dict
            # saved_root = "./saved_model/" + args.dataset + "/"
            if not os.path.exists(saved_root):
                os.mkdir(saved_root)
            torch.save(model.state_dict(), saved_root + str(model_num) + '.pth')
            # test_score = validate(args, model, testloader, multimask_output)
            # print('Epoch %d : test score : %f' % (epoch_num + 1, test_score))
        else:
            not_increase_num += 1
            print("valid_score not increase for {} epoch".format(not_increase_num))
        # if valid_score > best_performance:
        #     best_performance = valid_score
        #     save_mode_path = os.path.join(snapshot_path, 'best.pth')
        #     try:
        #         model.save_lora_parameters(save_mode_path)
        #     except:
        #         model.module.save_lora_parameters(save_mode_path)

        # test_score = validate(args, model, dice_loss, testloader, multimask_output)
        # logger.log({'info/test_score': test_score})

    # save_mode_path = os.path.join(snapshot_path, 'final.pth')
    # try:
    #     model.save_lora_parameters(save_mode_path)
    # except:
    #     model.module.save_lora_parameters(save_mode_path)
    # 加载 state_dict
    model_path = saved_root + str(model_num) + '.pth'
    # model_path = ("/media/wjy/D/python_code/model/"
    #               "cpc_sam/saved_model/celeb/best_model/355689_4.pth")
    model.load_state_dict(torch.load(model_path))
    # model.load_state_dict(torch.load(saved_root + str(model_num) + '.pth'))
    print("load model:", model_path)
    test_score, test_score_list = validate(args, model, testloader, multimask_output)
    print('test score : %f' % test_score)
    if test_score_list is not None:
        print("test_score_list:", test_score_list)

    return "Training Finished!"


if __name__ == '__main__':
    print('test')
