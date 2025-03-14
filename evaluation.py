"""
coding: utf-8
author: Lu Shiliang
date:2023-09
"""

import json
import os
import sys

import numpy as np
import torch
from torchvision import transforms
from transformers import ViTModel,SwinModel

from tqdm import tqdm
# import models.biformer as biformer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from Dataset.utils import read_data
from Dataset.MyDataset import ImageDataset
import matplotlib.pyplot as plt

# from Swin.swin_transformer import SwinTransformer

# from VIT.visual_transformer import vit_base_patch16_224

# from CropNet.CropNet_9 import MultiScaleSwin


from Network.PestNet_1 import MultiScaleSwin

if __name__ == '__main__':
    print("hello-1")
    # model = biformer.biformer_base(pretrained=False, nm_class=102)
    # model = SwinTransformer()
    # vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')  # 使用 ViTModel
     # 加载基础Swin模型（使用tiny版本作为示例）
    swin_model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
    
    # 创建多尺度模型
    model = MultiScaleSwin(swin_model, num_scales=3, num_classes=102)


    
    print("hello-2")
    weight_dict = torch.load('/home/zoujixiang/Pest_Recognition-main-test/Output37/checkpoint.pth')
    print("hello-3")
    model.load_state_dict(weight_dict["model"], strict=True)
    print("hello-4")


    model.to('cuda')
    model.eval()
    datadict = read_data('/home/zoujixiang/Pest_Recognition-main-test/IP102', 3)
    img_size = 224
    batch_size = 32
    data_transform = {
        "test": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    test_dataset = ImageDataset(images_path=datadict['test'][0],
                                images_class=datadict['test'][1],
                                transform=data_transform["test"])
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=4
                                              )
    predict_labels = []
    true_labels = []
    # sample_num = 0
    data_loader = tqdm(test_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        image, label = data
        for i in range(len(label)):
            true_labels.append(label[i].cpu().numpy())

        # sample_num += image.shape[0]
        
        pred = model(image.to('cuda'))
        # print(type(pred))
        pred_classes = torch.max(pred, dim=1)[1].cpu()
        for i in range(len(pred_classes)):
            predict_labels.append(pred_classes[i].cpu().numpy())

        macro = f1_score(true_labels, predict_labels, average='macro')
        weighted = f1_score(true_labels, predict_labels, average='weighted')
        acc = accuracy_score(true_labels, predict_labels)

        data_loader.desc = "[testing. F1-macro {:.3f},F1-weighted: {:.3f}, Acc: {:.3f}".format(macro, weighted, acc)

    print('macro:', f1_score(true_labels, predict_labels, average='macro'))
    print('weighted:', f1_score(true_labels, predict_labels, average='weighted'))
    print('Acc:', accuracy_score(true_labels, predict_labels))

    for i in range(len(true_labels)):
        true_labels[i] = int(true_labels[i])
        predict_labels[i] = int(predict_labels[i])

    with open('results11.txt','w') as f:
        f.write(str(true_labels))
        f.write('\n')
        f.write(str(predict_labels))

    json_path = 'class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    with open('class_indices_TRUENAME.json', "r") as f:
        class_Name = json.load(f)
    classes = []
    for k, i in enumerate(class_indict):
        classes.append(class_Name[i])
    print(classes)

    confusion = confusion_matrix(true_labels, predict_labels, normalize='true')
    plt.figure(figsize=(30, 28))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes, fontproperties='Times New Roman', fontsize=8)
    plt.yticks(indices, classes, fontproperties='Times New Roman', fontsize=8)
    plt.colorbar()
    plt.xlabel('Predict label', fontsize=24, family='Times New Roman')
    plt.xticks(rotation=90)
    plt.ylabel('True label', fontsize=24, family='Times New Roman')
    iters = np.reshape([[[i, j] for j in range(confusion.shape[0])] for i in range(confusion.shape[1])],
                       (confusion.size, 2))

    for i, j in iters:
        plt.text(j, i, '{:.3f}'.format(confusion[i, j]), va='center', ha='center', fontproperties='Times New Roman',
                fontsize=5)

    plt.savefig('Confusion_matrix_10.png', dpi=300)

    print(classification_report(true_labels, predict_labels, digits=4))

    sys.exit()