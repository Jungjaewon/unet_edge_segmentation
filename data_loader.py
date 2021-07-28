import os
import os.path as osp
import glob

from torch.utils import data
from torchvision import transforms as T
from PIL import Image


class DataSet(data.Dataset):

    def __init__(self, config, img_transform):
        self.img_transform = img_transform
        self.img_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR'], config['TRAINING_CONFIG']['MODE'])
        self.img_size = (config['MODEL_CONFIG']['IMG_SIZE'], config['MODEL_CONFIG']['IMG_SIZE'], 3)
        self.domain = config['TRAINING_CONFIG']['DOMAIN']

        if self.domain == 'ch':
            self.data_list = glob.glob(os.path.join(self.img_dir, '*.png'))
        else:
            self.data_list = glob.glob(os.path.join(self.img_dir, '*.jpg'))

        self.data_list = [x.split(os.sep)[-1].split('_')[0] for x in self.data_list]
        #self.data_list = list(set(self.data_list))

    def __getitem__(self, index):
        file_name = self.data_list[index]

        if self.domain == 'ch':
            image = Image.open(osp.join(self.img_dir, f'{file_name}_color.png')).convert('RGB')
            edge = Image.open(osp.join(self.img_dir, f'{file_name}_sketch.png')).convert('RGB')
        else:
            image = Image.open(osp.join(self.img_dir, f'{file_name}.jpg')).convert('RGB')
            edge = Image.open(osp.join(self.img_dir, f'{file_name}_edge.jpg')).convert('RGB')
        return self.img_transform(image), self.img_transform(edge)

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


def get_loader(config):

    img_transform = list()
    img_size = config['MODEL_CONFIG']['IMG_SIZE']

    img_transform.append(T.Resize((img_size, img_size)))
    img_transform.append(T.ToTensor())
    img_transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform = T.Compose(img_transform)

    dataset = DataSet(config, img_transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'],
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
