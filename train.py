import argparse
import os
import random
import time
import torch
from torch import nn
from data import data_helper
import torch.nn.functional as F
from models.resnet_domain import resnet18, resnet50
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.utils import AverageMeter
from utils.tools import *
import torch.autograd as autograd


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--target", default=2, type=int, help="Target")
    parser.add_argument("--device", type=int, default=6, help="GPU num")
    parser.add_argument("--time", default=0, type=int, help="train time")

    parser.add_argument("--eval", default=0, type=int, help="Eval trained models")
    parser.add_argument("--eval_model_path", default="/model/path", help="Path of trained models")

    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")

    parser.add_argument("--data_root", default="/home/Datasets/CV")
    parser.add_argument("--data", default="PACS")
    parser.add_argument("--val_perc", type=float, default=0.2, help="validation percentage")
    parser.add_argument("--result_path", default="./data/save/models/", help="")

    # data aug stuff
    parser.add_argument("--learning_rate", "-l", type=float, default=.002, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--gray_flag", default=1, type=int, help="whether use random gray")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.5, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")
    parser.add_argument("--network", choices=['resnet18', 'resnet50'], help="Which network to use",
                        default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float,
                        help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool,
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=True, type=bool, help="Use nesterov")
    return parser.parse_args()


def get_results_path(args):
    # Make the directory to store the experimental results
    base_result_path = args.result_path + "/" + args.data + "/"
    base_result_path += args.network
    base_result_path += "_lr" + str(args.learning_rate) + "_B" + str(args.batch_size)
    base_result_path += "/" + args.target + str(args.time) + "/"
    if not os.path.exists(base_result_path):
        os.makedirs(base_result_path)
    return base_result_path


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        if args.network == 'resnet18':
            model = resnet18(
                pretrained=True,
                device=device,
                classes=args.n_classes,
                domains=args.n_domains,
                network=args.network,
            )
        elif args.network == 'resnet50':
            model = resnet50(
                pretrained=True,
                device=device,
                classes=args.n_classes,
                domains=args.n_domains,
                network=args.network,
            )
        else:
            raise NotImplementedError("Not Implemented Network.")

        self.model = model.to(device)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args)
        self.target_loader = data_helper.get_target_dataloader(args)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset),
                                                           len(self.val_loader.dataset),
                                                           len(self.target_loader.dataset)))

        self.optimizer, self.scheduler = \
            get_optim_and_scheduler(model=model,
                                    network=args.network,
                                    epochs=args.epochs,
                                    lr=args.learning_rate,
                                    nesterov=args.nesterov)
        self.n_classes = args.n_classes
        self.n_domains = args.n_domains
        self.base_result_path = get_results_path(args)

        self.val_best = 0.0
        self.criterion = nn.CrossEntropyLoss()

    def _do_epoch(self, epoch=None):
        losses = AverageMeter()
        losses_l2 = AverageMeter()
        class_acc = AverageMeter()

        self.model.train()
        for it, ((_, image, target, domain), d_idx) in enumerate(self.source_loader):
            image = image.to(self.device)
            target = target.to(self.device)
            domain = domain.to(self.device)
            batch_size = image.size(0)

            # get predictions
            # aug_list = ['freq_dropout', 'freq_noise', 'freq_mixup']
            aug_list = ['spatial_dropout', 'freq_dropout', 'freq_noise', 'freq_mixup']
            logit, spatial_logit = self.model(image, labels=target, aug_mode=random.choice(aug_list))

            # calculate loss and optimize model
            loss = self.criterion(logit, target) 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update loss
            logit = logit.max(dim=1)[1]
            losses.update(loss.item(), batch_size)
            class_acc.update((logit==target).sum()/batch_size, 1)

            # print info
            if it%50==0 or it==len(self.source_loader)-1:
                print ('epoch: {}/{}, iter: {:3d} '.format(epoch, self.args.epochs, it), 
                       'loss: {:.4f} '.format(losses.avg), \
                       'acc: {:.2f} '.format(class_acc.avg*100))

        self.model.eval()
        with torch.no_grad():
            val_test_acc = []
            for phase, loader in self.test_loaders.items():
                class_acc, _ = self.do_test(phase, loader)
                val_test_acc.append(class_acc)
                self.results[phase][self.current_epoch] = class_acc
                print (phase, 'acc: ', format(class_acc*100, '.2f'))
            if val_test_acc[0] >= self.val_best:
                self.val_best = val_test_acc[0]
                self.save_model(mode="best")

    def do_training(self):
        self.results = {"val": torch.zeros(self.args.epochs), \
                        "test": torch.zeros(self.args.epochs)}
        
        for self.current_epoch in range(self.args.epochs):
            start_time = time.time()
            self._do_epoch(self.current_epoch)
            self.scheduler.step()
            end_time = time.time()
            print("Time for one epoch is " + str(format(end_time-start_time, '.0f')) + "s")
        
        self.save_model(mode="last")
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        line = "Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
        val_res.max(), test_res[idx_best], test_res.max(), idx_best)
        print(line)
        with open(self.base_result_path+"test.txt", "a") as f:
            f.write(line+"\n")
        return self.model

    def do_eval(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                class_acc, losses = self.do_test(phase, loader)
                result = phase + ": CELoss: " + str(format(losses.avg, '.4f')) \
                               + ", ACC: " + str(format(class_acc.avg, '.4f'))
                print(result)

    def do_test(self, phase, loader):
        class_acc = AverageMeter()
        losses = AverageMeter()
        for it, ((_, image, target, domain), _) in enumerate(loader):
            image = image.to(self.device)
            target = target.to(self.device)
            logit = self.model(image)[0]           
            loss = self.criterion(logit, target)
            logit = logit.max(dim=1)[1]
            class_acc.update(torch.sum(logit==target).item()/image.size(0))
            losses.update(loss.item(), image.size(0))
        return class_acc.avg, losses.avg

    def save_model(self, mode="best"):
        model_path = self.base_result_path + "models/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name = "model_" + mode + ".pth"
        torch.save({'state_dict': self.model.state_dict()}, 
                    os.path.join(model_path, model_name))


domain_map = {
    'PACS': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'OfficeHome': ['Art', 'Clipart', 'Product', 'RealWorld'],
    'VLCS': ["CALTECH", "LABELME", "PASCAL", "SUN"],
    'TerraInc': ['location_38', 'location_43', 'location_46', 'location_100']
}

classes_map = {
    'PACS': 7,
    'OfficeHome': 65,
    'VLCS': 5,
    'TerraInc': 10
}

val_perc_map = {
    'PACS': 0.2,
    'OfficeHome': 0.2,
    'VLCS': 0.2,
    'TerraInc': 0.2
}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_domain(name):
    if name not in domain_map:
        raise ValueError('Name of dataset unknown %s' %name)
    return domain_map[name]

def main():
    args = get_args()

    domain = get_domain(args.data)
    args.target = domain.pop(args.target)
    args.source = domain
    print("Target domain: {}".format(args.target))
    args.data_root = os.path.join(args.data_root, args.data)
    args.n_classes = classes_map[args.data]
    args.n_domains = len(domain)
    args.val_perc = val_perc_map[args.data]
    setup_seed(args.time)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(args, device)
    if args.eval:
        model_path = args.eval_model_path
        trainer.do_eval(model_path=model_path)
        return
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()


