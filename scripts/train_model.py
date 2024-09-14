import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset
import numpy as np

from glob import glob
import os
import timm
import cv2
import random
from tqdm import tqdm
from PIL import Image

from sklearn.metrics import accuracy_score, balanced_accuracy_score

np.random.seed(42)
torch.random.manual_seed(42)
random.seed(42)


# replace global_pool with 'max' to train the second model
GLOBAL_POOL = 'avg'
NUM_CLASSES = 9
BATCH_SIZE = 48

PATH_TO_DATASET = '../data/NCT-CRC-HE-100K/'
PATH_TO_TEST_DATASET = '../data/CRC-VAL-HE-7K/'


class Dataset(Dataset):
    def __init__(self, dir, aug=False):
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.aug = aug
        self.samples = [i for i in glob(os.path.join(dir, '**/*')) if os.path.isfile(i)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        class_name = os.path.basename(self.samples[idx]).split('-')[0]
        label = {
            'ADI': 0,
            'BACK': 1,
            'DEB': 2,
            'LYM': 3,
            'MUC': 4,
            'MUS': 5,
            'NORM': 6,
            'STR': 7,
            'TUM': 8,
        }

        img = cv2.imread(self.samples[idx], -1)[:, :, ::-1]
        img = np.float32(img) / 255.0

        if self.aug:
            if random.random() < 0.5:
                img = img[::-1]

            if random.random() < 0.5:
                img = img[:, ::-1]

            if random.random() < 0.3:
                if random.random() < 0.5:
                    img += np.random.normal(
                        0.0, np.random.uniform(0.01, 0.2),
                        (img.shape[0], img.shape[1], img.shape[2])
                    )
                else:
                    x = np.random.uniform(0.02, 0.15)
                    img += np.random.uniform(
                        -x, x,
                        (img.shape[0], img.shape[1], img.shape[2])
                    )

            if random.random() < 0.9:
                img += np.random.uniform(-0.15, 0.15, (1, 1, 3))
                img *= np.random.uniform(0.85, 1.15, (1, 1, 3))

            if random.random() < 0.3:
                kX, kY = np.random.randint(0, 4, 2) * 2 + 1
                if random.random() < 0.5:
                    img = cv2.GaussianBlur(img, (kX, kY), 0)
                else:
                    img = cv2.blur(img, (kX, kY))

        img = np.uint8(np.clip(img, 0, 1) * 255.0)
        img = self.transform(Image.fromarray(img))

        return img, label[class_name]


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')

    model = timm.create_model('efficientnet_b0.ra_in1k', pretrained=True, num_classes=NUM_CLASSES, in_chans=3, global_pool=GLOBAL_POOL)  
    model.to(device)

    train_dataset = Dataset(dir=PATH_TO_DATASET, aug=True)
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=False, drop_last=True)
    test_dataset = Dataset(dir=PATH_TO_TEST_DATASET, aug=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)

    print('Training...')

    optim = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.75, verbose=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc = 0.0

    for e in range(100):
        model.train()
        losses = []
        for (img, y_true) in tqdm(train_dataloader):
            y_pred = model(img.to(device))
            loss = loss_fn(y_pred, y_true.to(device))

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
        
        torch.save(model.state_dict(), f'model_last.pt')

        scheduler.step()
        model.eval()

        targets_array = []
        predictions_array = []
        test_losses = []
        with torch.no_grad():

            test_iter = iter(test_dataloader)
            for j in range(len(test_dataloader)):

                image, labels = next(test_iter)
                image = image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                predictions = model(image)
                
                loss = loss_fn(predictions, labels)
                test_losses.append(loss.item())

                _, predictions = torch.max(predictions.data, 1)

                predictions = predictions.detach().cpu().numpy()
                targets = labels.detach().cpu().numpy()

                for k in range(targets.shape[0]):

                    target = targets[k]
                    predicted = predictions[k]

                    targets_array.append(target)
                    predictions_array.append(predicted)

            acc = accuracy_score(targets_array, predictions_array)
            bacc = balanced_accuracy_score(targets_array, predictions_array)

            print(e)
            print('Train loss: ' + str(np.mean(losses)))
            print('Test loss: ' + str(np.mean(test_losses)))
            print('Accuracy: ' + str(acc))
            print('Balanced Accuracy: ' + str(bacc))

            if best_acc < acc:
                best_acc = acc
                torch.save(model.state_dict(), f'model_best.pt')
