import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random
import resnet_liver
import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image

HOSPITAL_NAMES = ["D", "F", "G", "N"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=0.001, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    
    parser.add_argument('--model_weight_path', type=str, default="/mnt/raid/tangzichen/Liver/resnet50-pre.pth", help='Pretrained model weight path')
    parser.add_argument('--root_dir', type=str, default="/mnt/raid/tangzichen/Liver/", help='Root directory for the liver folder.')

    args = parser.parse_args()
    return args

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2
    elif args.dataset == 'liver':
        n_classes = 2
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model+add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == 'moon':
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(args.model+add, args.out_dim, n_classes, net_configs)
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16,8]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    net = ResNet50_cifar10()
                elif args.dataset == 'liver' and args.model.startswith('resnet'):
                    if args.model == "resnet18":
                        net = resnet_liver.resnet18()
                    elif args.model == 'resnet34':
                        net = resnet_liver.resnet34()
                    elif args.model == 'resnet50':
                        net = resnet_liver.resnet50()
                    elif args.model == 'resnet101':
                        net = resnet_liver.resnet101()
                    elif args.model == 'resnet152':
                        net = resnet_liver.resnet152()
                    else:
                        print("Models for liver dataset not supported yet")
                        exit(1)
                        
                    if args.model_weight_path != None:
                        missing_keys, unexpected_keys = net.load_state_dict(torch.load(args.model_weight_path), strict=False)
                    # # 冻结部分权重
                    # for param in net.parameters():  #对于模型的每个权重，使其不进行反向传播，即固定参数
                    #     param.requires_grad = False
                    # for param in net.fc.parameters(): # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层fc
                    #     param.requires_grad = True

                    # change fc layer structure
                    in_channel = net.fc.in_features # 输入特征矩阵的深度。net.fc是所定义网络的全连接层
                    net.fc = nn.Linear(in_channel, 2)  # 类别个数
                    net.to(device)
                elif args.model == "vgg16":
                    net = vgg16()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu", ds_size=[], args=None, results_save_folder=''):
    # logger.info('Training network %s' % str(net_id))
    logger.info(f' ***  Training network {str(net_id)}: {HOSPITAL_NAMES[net_id]}  ***  ')
    
    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)  #大概率比SGD收敛的好！！！！！！
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        train_acc = 0.
        # epoch_acc_collector = []
        for step, data in enumerate(train_dataloader,start=0):
            # print(data)
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = criterion(logits, labels.to(device))
            pred = torch.max(logits, 1)[1]

            # print(pred)------
            train_correct = torch.max(logits, dim=1)[1]
            # train_correct = (pred == labels).sum()
            train_acc += (train_correct == labels.to(device)).sum().item()
            # train_acc += train_correct.item()
            loss.backward()
            optimizer.step()
            # print statistics
            train_loss += loss.item()
            # print train process
            rate = (step+1)/len(train_dataloader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
        
        # epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Local epoch: %d Loss: %f Acc: %f' % (epoch, train_loss/ds_size[0]*100, train_acc/ds_size[0]*100))
        # strr=str(epoch)+"  "+str(train_loss/ds[0]*100)+'    '+str(train_acc/dataset_size*100)
        # with open(Path(os.path.join(result_save_folder, 'train_loss.txt')).as_posix(), 'a') as f:
        #     f.write(str(strr) + '\n')

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    net.eval() # 控制训练过程中的Batch normalization
    eval_acc = 0.0  # accumulate accurate number / epoch
    eval_loss = 0.0
    with torch.no_grad():
        for val_data in test_dataloader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            loss = criterion(outputs, val_labels.to(device))
            eval_loss += loss.item()

            predict_y = torch.max(outputs, dim=1)[1]
            eval_acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = eval_acc / ds_size[1]
        # if val_accurate >= best_acc:
        #     best_acc = val_accurate
        #     torch.save(net.state_dict(), Path(os.path.join(results_save_folder, 'best_resNet50.pth')).as_posix())
        # print('[Local epoch %d] val_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, eval_loss / step, val_accurate))

        # save val results
        with open(Path(os.path.join(results_save_folder, 'val_accurate.txt')).as_posix(), 'a') as f:
            f.write(str(val_accurate) + '\n')
        with open(Path(os.path.join(results_save_folder, 'val_loss.txt')).as_posix(), 'a') as f:
            f.write(str(eval_loss / len(test_dataloader)) + '\n')

    # with open(Path(os.path.join(result_save_folder, 'best_acc.txt')).as_posix(), 'a') as f:
    #     f.write(str(best_acc) + '\n')
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc/ds_size[0]*100, val_accurate

def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)

def local_train_net(nets, selected, args, results_save_folder, test_dl = None, device="cpu"):
    avg_acc = 0.0
    train_ds_size_list = []
    for net_id, net in nets.items():
        # if net_id not in selected:
        #     continue
        image_dir_name = os.path.join(args.root_dir, f'Images_AP_all_{HOSPITAL_NAMES[net_id]}')
    
        # image_path = Path(os.path.join(root_dir, image_dir_name)).as_posix()
        # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
        # results_save_folder = os.path.join(args.results_save_dir, f'results_{image_dir_name}')
        client_results_save_folder = os.path.join(results_save_folder, f'results_Images_AP_all_{HOSPITAL_NAMES[net_id]}')
        if not os.path.exists(client_results_save_folder):
            os.makedirs(client_results_save_folder)
        # logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)
        
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_liver_dataloader(args.batch_size, 32, root_dir=args.root_dir, hospital_name = HOSPITAL_NAMES[net_id])
        train_ds_size_list.append(len(train_ds_local))
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(train_ds_local)))
        n_epoch = args.epochs
        
        print(f' ***  Training network {str(net_id)}: {HOSPITAL_NAMES[net_id]}  ***  ')
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, 
                                      device=device, ds_size=[len(train_ds_local), len(test_ds_local)], results_save_folder=client_results_save_folder)
        print(f' *** Training finished *** ')
        
        logger.info("net %d final test acc %f" % (net_id, testacc))
        print(f"net {net_id} {HOSPITAL_NAMES[net_id]} final test acc: {testacc}")
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= len(selected)
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, train_ds_size_list

def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map

if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'Log-%s' % (datetime.datetime.now().strftime("%m-%d-%H-%M"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    exp_time = "Exp-" + datetime.datetime.now().strftime("%m-%d-%H-%M")
    results_save_dir = os.path.join(os.getcwd(), 'results', exp_time)
    global_results_save_folder = os.path.join(results_save_dir, 'Global_results')
    if not os.path.exists(global_results_save_folder):
        os.makedirs(global_results_save_folder)
            
    n_classes = 2

    # train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
    #                                                                                     args.datadir,
    #                                                                                     args.batch_size,
    #                                                                                     32)
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_liver_dataloader(args.batch_size, 32, root_dir = args.root_dir, hospital_name = None)


    if args.alg == 'fedavg':
        logger.info("Initializing nets")
        # Initialize nets for each client
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, len(HOSPITAL_NAMES), args)
        # Initialize global server net
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        best_acc = 0.0
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            print(f' *** Round {str(round)} ***')
            selected = np.arange(args.n_parties)
            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, train_ds_size_list = local_train_net(nets, selected, args, results_save_folder=results_save_dir, test_dl = None, device=device)

            # update global model
            total_data_points = sum(train_ds_size_list)
            fed_avg_freqs = [train_ds_size_list[r] / total_data_points for r in selected]
            # fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)
            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            print('>> Global Model Train accuracy: %f' % train_acc)
            print('>> Global Model Test accuracy: %f' % test_acc)
            
            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save(global_model.state_dict(), Path(os.path.join(global_results_save_folder, 'best_global_resNet50.pth')).as_posix())
            # save val results
            with open(Path(os.path.join(global_results_save_folder, 'val_accurate.txt')).as_posix(), 'a') as f:
                f.write(str(test_acc) + '\n')


    # elif args.alg == 'local_training':
    #     logger.info("Initializing nets")
    #     nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
    #     arr = np.arange(args.n_parties)
    #     local_train_net(nets, arr, args, net_dataidx_map, test_dl = test_dl_global, device=device)

    # elif args.alg == 'all_in':
    #     nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, 1, args)
    #     n_epoch = args.epochs
    #     nets[0].to(device)
    #     trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer, device=device)

    #     logger.info("All in test acc: %f" % testacc)

   