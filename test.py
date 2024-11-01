import resnet_liver
from utils import *

def compute_val_acc(net, test_dataloader, criterion, val_ds_size, device='cpu'):
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
        val_accurate = eval_acc / val_ds_size
    return val_accurate

if __name__ == '__main__':
    device = "cuda:2"
    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     # transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(224),  # 长宽比例不变，将最小边缩放到256
                                   # transforms.CenterCrop(224),  # 再中心裁减一个224*224大小的图片
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}  # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    root_dir = "/mnt/raid/tangzichen/Liver"
    ann_txt_dir_val = os.path.join(root_dir, 'val_AP_All.txt')
    
    validate_dataset = MyDataset(root_dir = root_dir, ann_txt_dir = ann_txt_dir_val, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,batch_size=32, shuffle=False,num_workers=8)
    hospital_names = ["D", "F", "G", "N"]
    criterion = nn.CrossEntropyLoss().to(device)
    result_dict = {}
    # for hospital_name in hospital_names:
    #     print(f'Test on hospital: {hospital_name}')
    #     trained_weight_path = "/mnt/raid/tangzichen/Liver/results_Images_AP_all_" + hospital_name
    #     best_acc_path = os.path.join(trained_weight_path, 'best_resNet50.pth')
    #     net = resnet_liver.resnet50()
    #     # change fc layer structure
    #     in_channel = net.fc.in_features # 输入特征矩阵的深度。net.fc是所定义网络的全连接层
    #     net.fc = nn.Linear(in_channel, 2)  # 类别个数
    #     net.load_state_dict(torch.load(best_acc_path, map_location=device))
    #     net.to(device)
    #     acc = compute_val_acc(net, validate_loader, criterion, val_num, device=device)
    #     result_dict[hospital_name] = '{:.6f}'.format(acc)
    #     print(f'Hospital {hospital_name} acc: {acc}')

    best_acc_path = "/mnt/raid/tangzichen/Liver/results_Images_AP_all/best_resNet50.pth"
    net = resnet_liver.resnet50()
    # change fc layer structure
    in_channel = net.fc.in_features # 输入特征矩阵的深度。net.fc是所定义网络的全连接层
    net.fc = nn.Linear(in_channel, 2)  # 类别个数
    net.load_state_dict(torch.load(best_acc_path, map_location=device))
    net.to(device)
    acc = compute_val_acc(net, validate_loader, criterion, val_num, device=device)
    print('{:.6f}'.format(acc))
        
    # print(result_dict)