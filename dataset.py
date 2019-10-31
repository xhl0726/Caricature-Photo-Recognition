import os
import cv2

from torch.utils.data import Dataset
from torchvision import transforms


class WCDataset(Dataset):
    def __init__(self, dataset_path):
        training_file = os.path.join(
            dataset_path,
            'EvaluationProtocols',
            'FaceVerification',
            'UnRestricted',
            'UnRestrictedView1_DevTrain.txt'
        )
        #.replace('\\','/')
        #training_file=tf.replace('\\','/')
        with open(training_file) as f:
            self.class_num = int(f.readline())
            self.class_names = []
            self.images = []
            for i in range(self.class_num):
                words = f.readline().split()
                class_name = ' '.join(words[:-2])
                self.class_names.append(class_name)
                self.images += [
                    (os.path.join(dataset_path, 'OriginalImages', class_name, 'C%05d.jpg'%(j+1)), i) for j in range(int(words[-2]))
                ] + [
                    (os.path.join(dataset_path, 'OriginalImages', class_name, 'P%05d.jpg'%(j+1)), i) for j in range(int(words[-1]))
                ]
                # image not turn positive

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((116, 100)),
            transforms.RandomCrop((112, 96)),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: (img*255-127.5)/128.),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = cv2.imread(image_path, 1)
        assert image is not None, 'file %s dose not exist' % image_path

        return self.transform(image), label