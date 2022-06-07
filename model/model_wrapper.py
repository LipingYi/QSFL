import json
import numpy
import logging
import sys
import os
import torch
from sklearn.metrics import recall_score, accuracy_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchnet.meter import APMeter
from tqdm import tqdm
from model.yolo import Darknet
from utils.utils import *
from utils.datasets import *
from data.dataset import Dataset, TestDataset
from utils import array_tool as at
from utils.config import opt
from model import FasterRCNNVGG16
from model.faster_rcnn_trainer import FasterRCNNTrainer
from utils.eval_tool import eval_detection_voc
from model.femnist_cnn import FEMNIST_CNN
import numpy as np
from collections import Counter
import logging
logging.getLogger('engineio').setLevel(logging.CRITICAL)


sys.path.append("")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
torch.set_num_threads(4)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
numpy.random.seed(1234)


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


class Yolo(object):
    def __init__(self, task_config):
        self.task_config = task_config
        self.model_config = load_json(task_config['model_config'])
        print(self.model_config)
        if 'train' in self.task_config:
            self.dataset = ListDataset(self.task_config['train'],
                                       augment=True,
                                       multiscale=self.model_config['multiscale_training'])
            logging.info('load data')
            self.dataloader = DataLoader(self.dataset,
                                         batch_size=self.task_config['batch_size'],
                                         shuffle=True,
                                         num_workers=self.task_config['n_cpu'],
                                         collate_fn=self.dataset.collate_fn)
            # TODO: add a valset for validate
            self.testset = ListDataset(self.task_config['test'],
                                       augment=False,
                                       multiscale=False)

            self.test_dataloader = DataLoader(
                self.testset,
                batch_size=self.task_config['batch_size'],
                num_workers=1,
                shuffle=False,
                collate_fn=self.testset.collate_fn
            )

            self.train_size = self.dataset.__len__()
            print("train_size:", self.train_size)
            self.valid_size = self.testset.__len__()
        else:
            # add eval_set in serber for calculate rp"
            self.server_eval = ListDataset(self.task_config['server_eval'],
                                           augment=False,
                                           multiscale=False)

            self.server_eval_dataloader = DataLoader(
                self.server_eval,
                batch_size=1,
                num_workers=1,
                shuffle=False,
                collate_fn=self.server_eval.collate_fn
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = Darknet(self.model_config['model_def']).to(self.device)
        assert os.path.exists(self.model_config['pretrained_weights'])
        self.yolo.load_darknet_weights(self.model_config['pretrained_weights'])
        logging.info('model construct completed')
        self.best_map = 0
        self.optimizer = torch.optim.Adam(self.yolo.parameters())

    def get_weights(self):
        params = [param.data.cpu().numpy()
                  for param in self.yolo.parameters()]
        return params

    def set_weights(self, parameters):
        for i, param in enumerate(self.yolo.parameters()):
            param_ = torch.from_numpy(parameters[i]).cuda()
            param.data.copy_(param_)

    def train_one_epoch(self):
        """
        Return:
            total_loss: the total loss during training
            accuracy: the mAP
        """
        self.yolo.train()
        for batch_i, (_, imgs, targets) in enumerate(self.dataloader):
            batches_done = len(self.dataloader) * 1 + batch_i
            imgs = Variable(imgs.to(self.device))
            targets = Variable(targets.to(self.device), requires_grad=False)
            loss, outputs = self.yolo(imgs, targets)
            loss.backward()
            if batch_i % 10 == 0:
                print("step: {} | loss: {:.4f}".format(batch_i, loss.item()))
            if batches_done % self.model_config["gradient_accumulations"]:
                # Accumulates gradient before each step
                self.optimizer.step()
                self.optimizer.zero_grad()
        return loss.item()

    def eval(self, dataloader, yolo, test_num=10000):
        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        total_losses = list()
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
            # Extract labels
            print("Begin Prepare labels")
            labels += targets[:, 1].tolist()
            print("Prepare labels ok, to evaluate")
            # Rescale target
            targets = Variable(targets.to(self.device), requires_grad=False)

            imgs = Variable(imgs.type(Tensor), requires_grad=False)
            with torch.no_grad():
                loss, outputs = yolo(imgs, targets)
                outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.5)
                total_losses.append(loss.item())
            targets = targets.to("cpu")
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= int(self.model_config['img_size'])
            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=0.5)
        if len(sample_metrics) > 0:
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
        else:
            return 0.0, 0.0, 0.0
        total_loss = sum(total_losses) / len(total_losses)
        return total_loss, AP.mean(), recall.mean()

    def validate(self):
        """
        In the current version, the validate dataset hasn't been set, 
        so we use the first 500 samples of testing set instead.
        """
        print("run validation")
        return self.evaluate(500)

    def evaluate(self, test_num=10000):
        """
        Return:
            total_loss: the average loss
            accuracy: the evaluation map
        """
        total_loss, mAP, recall = self.eval(
            self.test_dataloader, self.yolo, test_num)
        return total_loss, mAP, recall


class FasterRCNN(object):
    """
    In fasterRCNN model, we only return the total loss, calculated from:
        rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss,
    and mAP@0.5
    """

    def __init__(self, task_config):
        self.model_config = load_json(task_config['model_config_file'])
        self.model_config['voc_data_dir'] = task_config['data_path']
        self.opt = opt
        self.opt.log_filename = task_config['log_filename']
        self.opt._parse(self.model_config)
        self.dataset = Dataset(self.opt)
        logging.info('load data')
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.model_config['batch_size'],
                                     shuffle=True,
                                     num_workers=self.opt.num_workers)
        # TODO: add a valset for validate
        self.testset = TestDataset(self.opt)
        self.test_dataloader = DataLoader(
            self.testset,
            batch_size=self.model_config['batch_size'],
            num_workers=self.opt.test_num_workers,
            shuffle=False,
            pin_memory=True
        )
        self.train_size = self.dataset.__len__()
        self.valid_size = self.testset.__len__()
        self.faster_rcnn = FasterRCNNVGG16()
        logging.info('model construct completed')
        self.trainer = FasterRCNNTrainer(
            self.faster_rcnn, self.opt.log_filename
        ).cuda()
        if self.opt.load_path:
            self.trainer.load(self.opt.load_path)
            logging.info('load pretrained model from %s' % self.opt.load_path)
        self.best_map = 0
        self.lr_ = self.opt.lr

    def get_weights(self):
        params = [param.data.cpu().numpy()
                  for param in self.faster_rcnn.parameters()]
        return params

    def set_weights(self, parameters):
        for i, param in enumerate(self.faster_rcnn.parameters()):
            param_ = torch.from_numpy(parameters[i]).cuda()
            param.data.copy_(param_)

    def train_one_epoch(self):
        """
        Return:
            total_loss: the total loss during training
            accuracy: the mAP
        """
        pred_bboxes, pred_labels, pred_scores = list(), list(), list()
        gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
        self.trainer.reset_meters()
        for ii, (img, sizes, bbox_, label_, scale, gt_difficults_) in \
                tqdm.tqdm(enumerate(self.dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            self.trainer.train_step(img, bbox, label, scale)
            if (ii + 1) % self.opt.plot_every == 0:
                sizes = [sizes[0][0].item(), sizes[1][0].item()]
                pred_bboxes_, pred_labels_, pred_scores_ = \
                    self.faster_rcnn.predict(img, [sizes])
                pred_bboxes += pred_bboxes_
                pred_labels += pred_labels_
                pred_scores += pred_scores_
                gt_bboxes += list(bbox_.numpy())
                gt_labels += list(label_.numpy())
                gt_difficults += list(gt_difficults_.numpy())

        return self.trainer.get_meter_data()['total_loss']

    def eval(self, dataloader, faster_rcnn, test_num=10000):
        pred_bboxes, pred_labels, pred_scores = list(), list(), list()
        gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
        total_losses = list()
        for ii, (imgs, sizes, gt_bboxes_, gt_labels_, scale, gt_difficults_) \
                in tqdm.tqdm(enumerate(dataloader)):
            img = imgs.cuda().float()
            bbox = gt_bboxes_.cuda()
            label = gt_labels_.cuda()
            sizes = [sizes[0][0].item(), sizes[1][0].item()]
            pred_bboxes_, pred_labels_, pred_scores_ = \
                faster_rcnn.predict(imgs, [sizes])
            losses = self.trainer.forward(img, bbox, label, float(scale))
            total_losses.append(losses.total_loss.item())
            gt_bboxes += list(gt_bboxes_.numpy())
            gt_labels += list(gt_labels_.numpy())
            gt_difficults += list(gt_difficults_.numpy())
            pred_bboxes += pred_bboxes_
            pred_labels += pred_labels_
            pred_scores += pred_scores_
            if ii == test_num: break

        result = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            use_07_metric=False)
        total_loss = sum(total_losses) / len(total_losses)
        return total_loss, result

    def validate(self):
        """
        In the current version, the validate dataset hasn't been set,
        so we use the first 500 samples of testing set instead.
        """
        print("run validation")
        return self.evaluate(500)

    def evaluate(self, test_num=10000):
        """
        Return:
            total_loss: the average loss
            accuracy: the evaluation map
        """
        total_loss, eval_result = self.eval(
            self.test_dataloader, self.trainer.faster_rcnn, test_num)
        return total_loss, eval_result['map'], eval_result['mrec']


class FEmnistDataset(Dataset):

    def __init__(self, path, clientID = None):

        self.data = json.load(open(path, 'r'))

    def __getitem__(self, index: int):
        y = self.data['y'][index]
        return torch.FloatTensor(self.data['x'][index]).view((1,28,28)), torch.LongTensor([y])


    def __len__(self):
        return len(self.data['x'])


def convert_tensor_to_one_hot(tensor, num_classes, batch_size):
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.FloatTensor(batch_size, num_classes)

    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, tensor.unsqueeze(-1), 1)

    return y_onehot


class FEmnistCNN(object):
    def __init__(self, task_config):
        self.task_config = task_config
        self.model_config = load_json(task_config['model_config'])
        print(self.model_config)
        if 'train' in self.task_config:
            self.dataset = FEmnistDataset(self.task_config['train'], self.task_config.get('clientID'))
            logging.info('load data')
            self.dataloader = DataLoader(self.dataset,
                                         batch_size=self.task_config['batch_size'],
                                         shuffle=True,
                                         num_workers=self.task_config['n_cpu'])
            # TODO: add a valset for validate
            self.testset = FEmnistDataset(self.task_config['test'])

            self.test_dataloader = DataLoader(
                self.testset,
                batch_size=self.task_config['batch_size'],
                num_workers=1,
                shuffle=False
            )

            self.train_size = self.dataset.__len__()
            print("train_size:", self.train_size)
            self.valid_size = self.testset.__len__()
        else:
            # add eval_set in serber for calculate rp"
            self.server_eval = FEmnistDataset(self.task_config['server_eval'])

            self.server_eval_dataloader = DataLoader(
                self.server_eval,
                batch_size=2,
                num_workers=1,
                shuffle=False
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FEMNIST_CNN().to(self.device)
        self.loss_func = CrossEntropyLoss().to(self.device)
        self.femnistcnn = []
        # assert os.path.exists(self.model_config['pretrained_weights'])
        # self.yolo.load_darknet_weights(self.model_config['pretrained_weights'])
        logging.info('model construct completed')
        self.best_map = 0
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_weights(self):
        params = [param.data.cpu().numpy()
                  for param in self.model.parameters()]
        return params

    def set_weights(self, parameters):
        for i, param in enumerate(self.model.parameters()):
            param_ = torch.from_numpy(parameters[i]).cuda()
            param.data.copy_(param_)

    def train_one_epoch(self):
        """
        Return:
            total_loss: the total loss during training
            accuracy: the mAP
        """
        self.model.train()

        total_loss = 0
        total_loss_len = 0

        for step, (x, y) in enumerate(self.dataloader):
            print('x is ', x.shape)
            print('y is ', y.shape)
            b_x = x.to(self.device)  # batch x
            # print('y', y)
            b_y = y.squeeze().to(self.device)
            # if len(b_y.numpy().shape) > 1:
            #     b_y = b_y.squeeze()  # batch y
            if b_y.cpu().numpy().shape.__len__() == 0:
                continue
            output = self.model(b_x).squeeze()  # cnn output
            loss = self.loss_func(output, b_y)  # cross entropy loss
            self.optimizer.zero_grad()  # clear gradients for this training setp
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()

            total_loss += loss.cpu().item()
            total_loss_len += 1

        total_loss = total_loss / total_loss_len  # 'test_loss'
        return total_loss


    def eval(self, dataloader, model, test_num=10000):
        total_loss = []
        total_map = []
        total_recall = []

        for x, y in dataloader:
            b_x = x.to(self.device)  # batch x
            b_y = y.squeeze().to(self.device)  # batch y
            test_y = b_y
            # print(f'shape:{b_y.cpu().numpy().shape.__len__()} ')
            if b_y.cpu().numpy().shape.__len__() == 0:
                continue
            test_output = model(b_x)  # cnn output
            loss = self.loss_func(test_output, b_y).cpu()  # cross entropy loss

            total_loss.append(loss.item())

            pred_y = torch.max(test_output, 1)[1].data.squeeze().cpu()

            recall = recall_score(test_y.cpu().data.numpy(), pred_y.data.numpy(), average='macro')
            accuracy = accuracy_score(test_y.cpu().data.numpy(), pred_y.cpu().data.numpy())
            total_recall.append(recall)
            total_map.append(accuracy)
        # print([type(i) for i in [np.mean(total_loss), np.mean(total_map), np.mean(total_recall)]])
        print(f'total_loss:{total_loss}, total_map: {total_map}, total_recall :{total_recall}')
        return np.mean(total_loss).tolist(), np.mean(total_map).tolist(), np.mean(total_recall).tolist()

    def validate(self):
        """
        In the current version, the validate dataset hasn't been set,
        so we use the first 500 samples of testing set instead.
        """
        print("run validation")
        return self.evaluate(500)

    def evaluate(self, test_num=10000):
        """
        Return:
            total_loss: the average loss
            accuracy: the evaluation map
        """
        total_loss, mAP, recall = self.eval(
            self.test_dataloader, self.model, test_num)
        return total_loss, mAP, recall




class Models:
    Yolo = Yolo
    FasterRCNN = FasterRCNN
    FEmnistCNN = FEmnistCNN
