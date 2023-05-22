# -*- coding: utf-8 -*-

import sys
import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import nltk
import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import math
from model import EncoderCNN, DecoderRNN
from data_loader import get_loader
from data_loader_val import get_loader as val_get_loader
from tqdm.notebook import tqdm
import torch.nn as nn
import json
from nlp_utils import clean_sentence, bleu_score
import matplotlib.pyplot as plt
from model import EncoderCNN, DecoderRNN

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

from google.colab import drive

mountPath = '/content/drive/'
drive.flush_and_unmount()
drive.mount(mountPath, force_remount=True)
projectPath = mountPath + 'MyDrive/image_captioning'
sys.path.append(projectPath)

"""Скачиваем val images """

!wget -P /content/ http://images.cocodataset.org/zips/val2014.zip
!unzip /content/val2014.zip -d /content/

"""Скачиваем train images"""

!wget -P /content/ http://images.cocodataset.org/zips/train2014.zip
!unzip /content/train2014.zip -d /content/

"""Скачиваем annotations"""

!wget -P /content/ http://images.cocodataset.org/annotations/annotations_trainval2014.zip
!unzip /content/annotations_trainval2014.zip -d /content/

"""Скачиваем train images"""

!wget -P /content/ http://images.cocodataset.org/zips/test2014.zip
!unzip /content/test2014.zip.1 -d /content/

batch_size = 128
vocab_threshold = 5
vocab_from_file = False
embed_size = 256
hidden_size = 512
num_epochs = 3
save_every = 1
print_every = 20
log_file = "training_log.txt"
cocoapi_dir = r"/content"

transform_train = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ]
)

data_loader = get_loader(
    transform=transform_train,
    mode="train",
    batch_size=batch_size,
    vocab_threshold=vocab_threshold,
    vocab_from_file=vocab_from_file,
    cocoapi_loc=cocoapi_dir,
)

vocab_size = len(data_loader.dataset.vocab)
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

criterion = (
    nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
)

params = list(decoder.parameters()) + list(encoder.embed.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)

total_step = math.ceil(len(data_loader.dataset) / data_loader.batch_sampler.batch_size)

print(total_step)

f = open(log_file, "w")

for epoch in range(1, num_epochs + 1):
    for i_step in range(1, total_step + 1):
        indices = data_loader.dataset.get_train_indices()
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler
        images, captions = next(iter(data_loader))
        images = images.to(device)
        captions = captions.to(device)

        decoder.zero_grad()
        encoder.zero_grad()

        features = encoder(images)
        outputs = decoder(features, captions)
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        loss.backward()
        optimizer.step()
        
        stats = (
            f"Epoch [{epoch}/{num_epochs}], Step [{i_step}/{total_step}], "
            f"Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):.4f}"
        )
        f.write(stats + "\n")
        f.flush()

        if i_step % print_every == 0:
            print("\r" + stats)

    if epoch % save_every == 0:
        torch.save(
            decoder.state_dict(), os.path.join("decoder-%d.pkl" % epoch)
        )
        torch.save(
            encoder.state_dict(), os.path.join("encoder-%d.pkl" % epoch)
        )
f.close()

torch.save(decoder.state_dict(), os.path.join('decoder-final.pkl'))
torch.save(encoder.state_dict(), os.path.join('encoder-final.pkl'))

"""EVAl"""

valPath = '/content'

transform_test = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ]
)

val_data_loader = val_get_loader(
    transform=transform_test, mode="valid", cocoapi_loc=valPath
)

encoder_file = "encoder-3.pkl"
decoder_file = "decoder-3.pkl"

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

encoder.load_state_dict(torch.load(os.path.join(encoder_file)))
decoder.load_state_dict(torch.load(os.path.join(decoder_file)))

encoder.eval()
decoder.eval()

pred_result = defaultdict(list)
for img_id, img in tqdm(val_data_loader):
    img = img.to(device)
    with torch.no_grad():
        features = encoder(img).unsqueeze(1)
        output = decoder.sample(features)
    sentence = clean_sentence(output, val_data_loader.dataset.vocab.idx2word)
    pred_result[img_id.item()].append(sentence)

with open(
    os.path.join(cocoapi_dir, "annotations/captions_val2014.json"), "r"
) as f:
    caption = json.load(f)

valid_annot = caption["annotations"]
valid_result = defaultdict(list)
for i in valid_annot:
    valid_result[i["image_id"]].append(i["caption"].lower())

list(valid_result.values())[:3]

list(pred_result.values())[:3]

bleu_score(true_sentences=valid_result, predicted_sentences=pred_result)

cocoapi_dir = r"/content"

transform_test = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ]
)

data_loader = get_loader(transform=transform_test, mode="test", cocoapi_loc=cocoapi_dir)

orig_image, image = next(iter(data_loader))
print(orig_image.shape, image.shape, np.squeeze(orig_image).shape)
plt.imshow(np.squeeze(orig_image))
plt.title("example image")
plt.show()

encoder_file = "encoder-3.pkl"
decoder_file = "decoder-3.pkl"

embed_size = 256
hidden_size = 512

vocab_size = len(data_loader.dataset.vocab)

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

encoder.eval()
decoder.eval()

encoder.load_state_dict(torch.load(os.path.join("/content", encoder_file)))
decoder.load_state_dict(torch.load(os.path.join("/content", decoder_file)))

encoder.to(device)
decoder.to(device)

image = image.to(device)
features = encoder(image).unsqueeze(1)

print(features.shape)
output = decoder.sample(features)
print("example output:", output)

sentence = clean_sentence(output, data_loader.dataset.vocab.idx2word)

def get_prediction(idx2word, i=0, save=False):
    orig_image, image = next(iter(data_loader))
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output, idx2word)

    ax = plt.axes()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    plt.imshow(np.squeeze(orig_image))
    plt.xlabel(sentence, fontsize=12)
    if save:
        plt.savefig(f"/content/samples/sample_{i:03}.png", bbox_inches="tight")
    plt.show()

for i in range(20):
    get_prediction(data_loader.dataset.vocab.idx2word, i=i)