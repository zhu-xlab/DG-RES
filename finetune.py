# from torch.utils.tensorboard import SummaryWriter
import argparse
from torch import nn
import torch.nn.functional as F
from data import data_helper
import torch.fft as fft
from models.resnet_domain import resnet18, resnet50
import os
import random
import time
from utils.tools import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=int, default=6, help="GPU num")
    parser.add_argument("--time", default=0, type=int, help="train time")

    parser.add_argument("--eval", default=0, type=int, help="Eval trained models")
    parser.add_argument("--eval_model_path", default="/model/path", help="Path of trained models")

    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")

    parser.add_argument("--data", default="PACS")
    parser.add_argument("--data_root", default="/home/Datasets/CV/")
    parser.add_argument("--target", type=str, help="Target", default='photo')
    parser.add_argument("--class_dict", default=None, type=list)

    parser.add_argument("--domain_flag", default=1, type=int, help="whether use domain discriminator.")
    parser.add_argument("--result_path", default="./data/save/models/", help="")

    # data aug stuff
    parser.add_argument("--learning_rate", "-l", type=float, default=.002, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--gray_flag", default=1, type=int, help="whether use random gray")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--network", choices=['resnet18', 'resnet50'], help="Which network to use",
                        default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float,
                        help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=True, type=bool, help="Use nesterov")

    return parser.parse_args()


def get_results_path(args):
    # Make the directory to store the experimental results
    base_result_path = args.result_path + "/" + args.data + "/"
    base_result_path += args.network
    base_result_path += "_lr" + str(args.learning_rate) + "_B64"
    base_result_path += "/" + args.target + str(args.time) + "/models" + '/model_best.pth' 
    return base_result_path

def center_crop_feature(feature_tensor, crop_height=16, crop_width=16):
    height, width = feature_tensor.size()
    center_h = height // 2
    center_w = width // 2
    cropped_features = feature_tensor[center_h-crop_height:center_h+crop_height, \
                                      center_w-crop_width:center_w+crop_width]
    return cropped_features

def norm_weights(weights):
    def calibrate_weight(data):
        sorted_data, indices = torch.sort(data)
        rank = torch.linspace(data.min(), data.max(), len(sorted_data))  # 数据在原分布中的排名
        cdf_values = (indices.float() + 1) / len(sorted_data)  # 原分布的累积分布函数值
        transformed_data = torch.lerp(torch.zeros_like(sorted_data), torch.ones_like(sorted_data), cdf_values).to(self.device)
        return transformed_data
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    # weights = calibrate_weight(weights)
    return weights

class FineTuner:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.network = args.network
        if self.network == 'resnet18':
            model = resnet18(
                pretrained=True,
                device=device,
                classes=args.n_classes,
                domains=args.n_domains,
                network=args.network,
            )
        elif self.network == 'resnet50':
            model = resnet50(
                pretrained=True,
                device=device,
                classes=args.n_classes,
                domains=args.n_domains,
                network=args.network,
            )
        else:
            raise NotImplementedError("Not Implemented Network.")

        self.base_result_path = get_results_path(args)
        print ('loading checkpoint of {} ...'.format(self.base_result_path))
        checkpoint = torch.load(self.base_result_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model = model.to(device)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args)
        self.target_loader = data_helper.get_target_dataloader(args, shuffle=False)
        self.len_dataloader = len(self.source_loader)
        print ("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset),
                                                            len(self.val_loader.dataset),
                                                            len(self.target_loader.dataset)))

        self.args = args
        self.n_domains = args.n_domains
        self.n_classes = args.n_classes
        self.domain_flag = args.domain_flag
        self.class_dict = args.class_dict
        self.criterion = nn.CrossEntropyLoss()

        self.dims = [64, 128, 256, 512] if self.network=='resnet18' else [256, 512, 1024, 2048]
        self.embedding_dict = {'feat': torch.zeros((0, self.dims[-1])).to(self.device), 
                               'class': torch.tensor([]).to(self.device), 
                               'domain': torch.tensor([]).to(self.device)}

    def print_confusion_matrix(self, cm, class_names, normalize=True):
        if normalize:
            normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        else:
            normalized_cm= cm

        # 打印表头
        header = f"Predicted:      "
        for name in class_names:
            header += f"{name:^10}"
        print(header)        
        for i in range(self.n_classes):
            row = f"Actual {class_names[i]:<9}"
            for j in range(self.n_classes):
                row += f"{normalized_cm[i, j]:^10.3f}"
            print(row)

        print ('overall accuracy: {:.2f}'.format(100 * np.trace(cm) / cm.sum()))
        print ()

    def calc_confusion_matrix(self):
        self.model.eval()
        confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        # calculate embedding
        with torch.no_grad():
            for it, ((_, image_t, class_t, domain_t), _) in enumerate(self.target_loader):
                image_t = image_t.to(self.device)
                pred_t = self.model(x=image_t).max(dim=1)[1] 
                for t, p in zip(class_t.view(-1), pred_t.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        self.print_confusion_matrix(confusion_matrix, self.class_dict)

    def visualize_attn_map(self):
        def reverse_image_transform(images):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            inverse_normalize = transforms.Compose([
                transforms.Normalize(mean=[0, 0, 0], std=[1 / std[0], 1 / std[1], 1 / std[2]]),  # 逆标准化
                transforms.Normalize(mean=[-mean[0], -mean[1], -mean[2]], std=[1, 1, 1]),  # 逆均值化
            ])
            restored_images = torch.stack([inverse_normalize(image) for image in images])
            restored_images = (torch.clamp(restored_images, 0, 1) * 255).int()
            return restored_images.permute(0,2,3,1).cpu().data.numpy().astype(np.uint8)

        def normalize_cam(feat):
            # feat: H x W 
            h, w = feat.size()
            feat = feat.view(h*w)     # (H x W)
            cam = feat / feat.sum()
            cam = cam.view(h, w).cpu().data.numpy()
            cam = ((cam - np.min(cam)) / (np.max(cam) - np.min(cam)) * 255).astype(np.uint8)
            return cam

        def get_cam(x, labels):
            with torch.no_grad():
                h, w = x.size()[-2:]
                x = self.model.conv1(x)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)
                fc_weights = self.model.classifier.weight.data  
                conv_weights = fc_weights.view(fc_weights.size(0), fc_weights.size(1), 1, 1)   
                logit = F.conv2d(x, conv_weights)

                cam = logit.mean(dim=1)
                for i in range(labels.size(0)):
                    cam[i] = logit[i, labels[i]]
                    cam[i] = (cam[i] - cam[i].min()) / (cam[i].max() - cam[i].min())

                # 对张量进行 softmax 操作，将其转换为概率分布
                probabilities = F.softmax(logit, dim=1)
                entropy = probabilities.sum(dim=1)
                for i in range(labels.size(0)):
                    entropy[i] = -probabilities[i, labels[i]] * torch.log2(probabilities[i, labels[i]] + 1e-12)  
                entropy = 1 - (entropy - entropy.min()) / (entropy.max() - entropy.min())

                return entropy

        self.model.eval()
        for it, ((_, image_s, class_s, domain_s), _) in enumerate(self.source_loader):
            if it*self.args.batch_size > 200:
                break
            image_s = image_s.to(self.device)
            pred_s = self.model(image_s)[0]
            pred_s = pred_s.max(dim=1)[1]                 
            cam_s = get_cam(image_s, class_s)
            reverse_image_s = reverse_image_transform(image_s)    
            h, w = reverse_image_s.shape[1:3]
            for i in range(image_s.size(0)):
                cam = normalize_cam(cam_s[i])
                cam = cv2.resize(cam.astype(np.uint8), (h,w), interpolation=cv2.INTER_LINEAR)[:,:,None]
                cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
                fused_image = (cam*0.4 + reverse_image_s[i]*0.6).astype(np.uint8)
                save_path = './CAMs/source_{}_label_{}_predicted_{}.png'\
                    .format(str(it*self.args.batch_size + i), self.class_dict[class_s[i]], self.class_dict[pred_s[i]])
                cv2.imwrite(save_path, fused_image)
                print (save_path)

        for it, ((_, image_t, class_t, domain_t), _) in enumerate(self.target_loader):
            if it*self.args.batch_size > 100:
                break
            image_t = image_t.to(self.device)
            class_t = class_t.to(self.device)
            image_t.requires_grad = True
            pred_t = self.model(image_t)[0]
            pred_t = pred_t.max(dim=1)[1]                 

            cam_t = get_cam(image_t, class_t)
            reverse_image_t = reverse_image_transform(image_t)            
            h, w = reverse_image_t.shape[1:3]
            for i in range(image_t.size(0)):
                if pred_t[i] != class_t[i]:
                    cam = normalize_cam(cam_t[i])
                    cam = cv2.resize(cam.astype(np.uint8), (h,w), interpolation=cv2.INTER_LINEAR)[:,:,None]
                    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
                    fused_image = (cam*0.4 + reverse_image_t[i]*0.6).astype(np.uint8)
                    save_path = './CAMs/target_{}_label_{}_predicted_{}.png'\
                        .format(str(it*self.args.batch_size + i), self.class_dict[class_t[i]], self.class_dict[pred_t[i]])
                    cv2.imwrite(save_path, fused_image)
                    print (save_path)

    def calc_embedding(self):
        self.model.eval()
        magnitude_layer = 'conv1'
        magnitude_size = 112
        crop_size = 8
        magnitude = {str(i):torch.zeros((0, magnitude_size, magnitude_size)).to(self.device) \
                        for i in range(self.n_domains+1)}

        # calculate embedding
        with torch.no_grad():
            # source loader
            for it, ((_, data_s, class_s, domain_s), d_idx) in enumerate(self.source_loader):
                data_s = data_s.to(self.device)
                class_s = class_s.to(self.device)
                domain_s = domain_s.to(self.device)
                feat_s, magni_s = self.model.get_features(x=data_s, magnitude_layer=magnitude_layer)    
                feat_s = self.model.pooling(feat_s)
                for i in range(domain_s.size(0)):
                    domain_i = domain_s[i].int().cpu().numpy()
                    magnitude[str(domain_i)] = torch.cat((magnitude[str(domain_i)], magni_s[i].unsqueeze(dim=0)), dim=0) 
                self.embedding_dict['feat'] = torch.cat((self.embedding_dict['feat'], feat_s), dim=0)
                self.embedding_dict['class'] = torch.cat((self.embedding_dict['class'], class_s), dim=0)
                self.embedding_dict['domain'] = torch.cat((self.embedding_dict['domain'], domain_s), dim=0)
                if it > 30:
                    break

            # target loader
            for it, ((_, data_t, class_t, domain_t), d_idx) in enumerate(self.target_loader):
                data_t = data_t.to(self.device)
                class_t = class_t.to(self.device)
                domain_t = domain_t.to(self.device)
                feat_t, magni_t = self.model.get_features(x=data_t, magnitude_layer=magnitude_layer)     
                feat_t = self.model.pooling(feat_t)
                for i in range(domain_t.size(0)):
                    domain_i = str(domain_t[i].cpu().numpy())
                    magnitude[domain_i] = torch.cat((magnitude[domain_i], magni_t[i].unsqueeze(dim=0)), dim=0) 
                self.embedding_dict['feat'] = torch.cat((self.embedding_dict['feat'], feat_t), dim=0)
                self.embedding_dict['class'] = torch.cat((self.embedding_dict['class'], class_t), dim=0)
                self.embedding_dict['domain'] = torch.cat((self.embedding_dict['domain'], domain_t), dim=0)
                if it > 10:
                    break

            # for i in range(self.n_domains+1):
            #     domain = str(i)
            #     domain_name = self.args.source[int(domain)] if i < self.n_domains else self.args.target
            #     magnitude[domain] = magnitude[domain].mean(dim=0).clip(max=255).clip(min=0) 
            #     print (magnitude[domain].shape, crop_size)
            #     magnitude[domain] = center_crop_feature(magnitude[domain], crop_size, crop_size)
            #     # magnitude[domain] = (magnitude[domain] - magnitude[domain].min()) \
            #     #                   / (magnitude[domain].max() - magnitude[domain].min()) * 255
            #     magnitude[domain] = (magnitude[domain]).cpu().numpy().astype(np.uint8)
            #     resized_magnitude = cv2.resize(magnitude[domain], (512, 512))
            #     resized_magnitude = cv2.applyColorMap(resized_magnitude, cv2.COLORMAP_JET)
            #     cv2.imwrite('magnitudes/magnitude_domain_{}_{}_{}.jpg'.format(self.args.network, magnitude_layer, domain_name), resized_magnitude)

        self.embedding_dict['feat'] = self.embedding_dict['feat'].detach().cpu().numpy()

    def visualize_tsne(self, title="t-SNE Visualization"):
        tsne = TSNE(n_components=2, random_state=0)
        embedded_data = tsne.fit_transform(self.embedding_dict['feat'])

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']        
        shapes = ['o', 's', '^', 'x', 'v', 'v', '>']
        for i in range(embedded_data.shape[0]):
            class_i = int(self.embedding_dict['class'][i])
            domain_i = int(self.embedding_dict['domain'][i])
            color = colors[class_i]
            shape = shapes[domain_i]
            plt.scatter(embedded_data[i, 0], embedded_data[i, 1], \
                        c=color, s=3, marker=shape)

        plt.title(title)
        # plt.legend()
        plt.savefig('./tSNEs/{}_{}_{}_tSNE_Target.png'\
                    .format(self.network, self.args.data, self.args.target), dpi=1000) 


domain_map = {
    'PACS': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'OfficeHome': ['Art', 'Clipart', 'Product', 'RealWorld'],
    'VLCS': ["CALTECH", "LABELME", "PASCAL", "SUN"]
}

classes_map = {
    'PACS': 7,
    'OfficeHome': 65,
    'VLCS': 5,
}

classes_dict = {
    'PACS': ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'],
    'OfficeHome': ['Alarm Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles',
                   'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk Lamp', 'Drill', 'Eraser', 'Exit Sign', 'Fan',
                   'File Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard',
                   'Knives', 'Lamp Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan',
                   'Paper Clip', 'Pen', 'Pencil', 'Postit Notes', 'Printer', 'Push Pin', 'Radio', 'Refrigerator', 'ruler',
                   'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone',
                   'Toothbrush', 'Toys', 'Trash Can', 'TV', 'Webcam'],
    'VLCS': ['bird', 'car', 'chair', 'dog', 'person'],
}

val_perc_map = {
    'PACS': 0.2,
    'OfficeHome': 0.2,
    'VLCS': 0.2,
}

def get_domain(name):
    if name not in domain_map:
        raise ValueError('Name of dataset unknown %s' %name)
    return domain_map[name]

def main():
    args = get_args()

    domain = get_domain(args.data)
    domain.remove(args.target)
    args.source = domain
    args.n_domains = len(domain)
    print("Target domain: {}".format(args.target))
    args.data_root = os.path.join(args.data_root, "PACS") \
            if "PACS" in args.data else os.path.join(args.data_root,
                                                     args.data)
    args.n_classes = classes_map[args.data]
    args.val_perc = val_perc_map[args.data]
    args.class_dict = classes_dict[args.data]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tuner = FineTuner(args, device)
    # tuner.calc_embedding()
    # tuner.visualize_tsne()
    tuner.visualize_attn_map()
    # tuner.calc_confusion_matrix()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()


