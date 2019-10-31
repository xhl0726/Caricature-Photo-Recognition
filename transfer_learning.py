import os
import shutil

import tensorboardX
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import WCDataset
from net_sphere import AngleLinear, AngleLoss, sphere20a, LSoftmaxLinear
from utils import Timer, get_config, get_model_list


def get_net(model_path, class_num):
    net = sphere20a()
    net.load_state_dict(torch.load(model_path))
    net.fc6 = AngleLinear(512, class_num)

    net.classnum = class_num

    for name, param in net.named_parameters():
        param.requires_grad = name.startswith('conv4') or name.startswith('relu4') or name.startswith('fc')  # or name.startswith('stn')

    return net


def get_optimizer(net, hyperparameters):
    return optim.SGD(
        [param for param in net.parameters() if param.requires_grad],
        lr=hyperparameters['lr'],
        momentum=hyperparameters['momentum'],
        weight_decay=hyperparameters['weight_decay'],
    )


def get_scheducer(optimizer, hyperparameters, last_epoch=-1):
    return optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=hyperparameters['milestones'],
        gamma=hyperparameters['gamma'],
        last_epoch=last_epoch
    )


def train(net, dataloader, criterion, optimizer, scheducer, hyperparameters, writer, checkpoint_directory, iteration):
    net.cuda()
    net.train()
    criterion = criterion.cuda()

    if iteration == -1:
        iteration = 0
    while True:
        total_count, accpeted_count = 0, 0
        
        with Timer("Elapsed time in update: %f"):
            for images, labels in dataloader:
                images, labels = images.cuda(), labels.cuda()

                scheducer.step()  # ??
                iteration += 1
                
                optimizer.zero_grad()
                outputs = net(images, labels)#返回每组P和C的特征提取结果
                loss = criterion(outputs, labels)#
                loss.backward()
                optimizer.step()

                total_count += len(images)
                accpeted_count += int(torch.sum(labels == torch.argmax(outputs[0], dim=1)))  #

                if iteration % hyperparameters['log_iter'] == 0:
                    writer.add_scalar('loss', float(loss), iteration)
                    writer.add_scalar('accurency', accpeted_count / total_count, iteration)
                    print("Iteration: %08d/%08d" % (iteration, hyperparameters['max_iter']))

                if iteration % hyperparameters['snapshot_save_iter'] == 0:
                    torch.save(net.state_dict(), os.path.join(checkpoint_directory, '%08d.pth' % iteration))

                if iteration >= hyperparameters['max_iter']:
                    return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/init.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--myGpu', default='0', help='GPU Number')
    opts = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.myGpu

    config = get_config(opts.config)
    
    # Setup logger and output foders
    # from git import Repo
    # repo = Repo('.')
    model_name = '%s_%s' % (os.path.splitext(os.path.basename(opts.config))[0], str('original_dataset'))  # init_..
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))  # ./logs......
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    os.makedirs(checkpoint_directory, exist_ok=True)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

    dataloader = DataLoader(
        WCDataset(config['dataset_path']),
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=config['num_workers'],
    )

    if opts.resume:  # opts.resume=False
        last_model_name = get_model_list(checkpoint_directory)
        iteration = int(last_model_name[:-4])
        net = get_net(last_model_name, dataloader.dataset.class_num)
        optimizer = get_optimizer(net, config)
        scheducer = get_scheducer(optimizer, config, iteration)
        print('Resume from iteration %d' % iteration)
    else:
        iteration = 0
        net = get_net(config['weight_path'], dataloader.dataset.class_num)
        optimizer = get_optimizer(net, config)
        scheducer = get_scheducer(optimizer, config)

    criterion = AngleLoss()
    # criterion = nn.CrossEntropyLoss()
    train(net, dataloader, criterion, optimizer, scheducer, config, train_writer, checkpoint_directory, iteration)
