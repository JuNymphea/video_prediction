import cv2
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import io
import ujson as json
import nori2 as nori


os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KittiTrainDataset_nori(Dataset):
    def __init__(self, args):
        self.args = args
        self.train_data = []
        nori_path = 's3://huxiaotao/nori/kitti_train.nori/kitti_train.json'
        with nori.smart_open(nori_path) as f:
            for data in json.load(f):
                self.train_data.append(data)
        self.nf = nori.Fetcher()

    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, index):
        imgs = self.get_data(index)             # (N)(H, W, C)
        imgs = self.aug_seq(imgs, 256, 256)     # (N)(256, 256, C)
        num_frames = len(imgs)  # N
        ' --- rotate'
        if random.randint(0, 1):
            for i in range(num_frames):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(0, 1):
            for i in range(num_frames):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
        elif random.randint(0, 1):
            for i in range(num_frames):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        ' --- channel reverse'
        if random.uniform(0, 1) < 0.5:
            for i in range(num_frames):
                imgs[i] = imgs[i][:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(num_frames):
                imgs[i] = imgs[i][::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(num_frames):
                imgs[i] = imgs[i][:, ::-1]
        ' --- (H, W, C) -> (C, H, W)'
        for i in range(num_frames):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)     # (N)(C, 224, 224)
        return torch.stack(imgs, 0)     # (N, C, 224, 224)

    def aug_seq(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs

    def get_data(self, index):
        data = self.train_data[index]
        imgs = []
        for i in range(9):
            imgs.append(
                cv2.imdecode(np.frombuffer(io.BytesIO(self.nf.get(data[i])).getbuffer(), np.uint8), cv2.IMREAD_COLOR)
            )
        ' --- vis'
        # os.makedirs(f'outputs/{index}', exist_ok=True)
        # if self.args.local_rank in [-1, 0]:
        #     for i in range(9):
        #         print(index)
        #         cv2.imwrite(
        #             f'outputs/{index}/{i}.png', cv2.imdecode(np.frombuffer(io.BytesIO(self.nf.get(data[i])).getbuffer(), np.uint8), cv2.IMREAD_COLOR)
        #         )
        return imgs


class KittiValDataset(Dataset):
    def __init__(self, args):
        self.val_data = []
        self.video_path = '/data/dmvfn/data/KITTI/test'
        self.video_data = sorted(os.listdir(self.video_path))
        for i in self.video_data:
            self.val_data.append(os.path.join(self.video_path, i))
        self.val_data = sorted(self.val_data)

    def __len__(self):
        return len(self.val_data)
        
    def getimg(self, index):
        data = self.val_data[index]
        img_list = sorted(os.listdir(data))
        imgs = []
        for i in range(2, 9):
            im = cv2.imread(os.path.join(data, img_list[i]))
            imgs.append(im)
        return imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        name = self.video_data[index]
        length = len(imgs)
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)
        return torch.stack(imgs, 0), name   # n, c, h, w


class VimeoTrainDataset_nori(Dataset):
    def __init__(self, args):
        self.args = args
        self.train_data = []
        nori_path = 's3://huxiaotao/nori/vimeo_septuplet.nori/vimeo_septuplet_train.json'
        with nori.smart_open(nori_path) as f:
            for data in json.load(f):
                self.train_data.append(data)
        self.nf = nori.Fetcher()

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        imgs = self.get_data(index)             # (N)(H, W, C)
        imgs = self.aug_seq(imgs, 224, 224)     # (N)(224, 224, C)
        num_frames = len(imgs)  # N

        ' --- rotate'
        if random.randint(0, 1):
            for i in range(num_frames):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(0, 1):
            for i in range(num_frames):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
        elif random.randint(0, 1):
            for i in range(num_frames):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        ' --- channel reverse'
        if random.uniform(0, 1) < 0.5:
            for i in range(num_frames):
                imgs[i] = imgs[i][:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(num_frames):
                imgs[i] = imgs[i][::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(num_frames):
                imgs[i] = imgs[i][:, ::-1]
        
        ' --- (H, W, C) -> (C, H, W)'
        for i in range(num_frames):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)     # (N)(C, 224, 224)
        return torch.stack(imgs, 0)     # (N, C, 224, 224)

    def aug_seq(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs

    def get_data(self, index):
        data = self.train_data[index]
        imgs = []
        for i in range(7):
            imgs.append(
                cv2.imdecode(np.frombuffer(io.BytesIO(self.nf.get(data[i])).getbuffer(), np.uint8), cv2.IMREAD_COLOR)
            )
        return imgs
        ' --- vis'
        # os.makedirs(f'outputs/{index}', exist_ok=True)
        # if self.args.local_rank in [-1, 0]:
        #     for i in range(7):
        #         print(index)
        #         cv2.imwrite(
        #             f'outputs/{index}/{i}.png', cv2.imdecode(np.frombuffer(io.BytesIO(self.nf.get(data[i])).getbuffer(), np.uint8), cv2.IMREAD_COLOR)
        #         )            


class VimeoValDataset(Dataset):
    def __init__(self, args):
        self.train_data = []
        with nori.smart_open('/data/vimeo_test.json') as f:
            for data in json.load(f):
                self.train_data.append(data)
        self.nf = nori.Fetcher()
    
    def __len__(self):
        return len(self.train_data)       

    def __getitem__(self, index):
        imgs = self.get_data(index)
        for i in range(len(imgs)):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)     # (C, H, W)
        return torch.stack(imgs, 0), torch.stack(imgs, 0)   # (N, C, H, W)

    def get_data(self, index):
        data = self.train_data[index]
        imgs = []
        for i in range(7):
            imgs.append(
                cv2.imdecode(np.frombuffer(io.BytesIO(self.nf.get(data[i])).getbuffer(), np.uint8), cv2.IMREAD_COLOR)
            )
        return imgs


class UCFTrainDataset(Dataset):
    def __init__(self):
        self.path = './data/ucf101_jpeg/jpegs_256/'
        self.train_data = []
        with open(os.path.join('/home/huxiaotao/trainlist01.txt')) as f:
            for line in f:
                video_dir = line.rstrip().split('.')[0]
                video_name = video_dir.split('/')[1]
                self.train_data.append(video_name)

    def __len__(self):
        return len(self.train_data)

    def aug_seq(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs

    def getimg(self, index):
        video_path = self.train_data[index]
        frame_list = sorted(os.listdir(os.path.join(self.path, video_path)))
        n = len(frame_list)
        max_time = 5 if 5 <= n/10 else int(n/10)
        time_step = np.random.randint(1, max_time + 1)#1, 2, 3, 4, 5
        frame_ind = np.random.randint(0, n-9*time_step)
        frame_inds = [frame_ind+j*time_step for j in range(10)]
        imgs = []
        for i in frame_inds:
            im = cv2.imread(os.path.join(os.path.join(self.path, video_path), frame_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        imgs = self.aug_seq(imgs, 256, 256)
        length = len(imgs)
        if random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, ::-1]
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)

        return torch.stack(imgs, 0)#n, c, h , w


class DavisValDataset(Dataset):
    def __init__(self):
        self.val_data = []
        self.video_path = './data/DAVIS/'
        self.video_data = sorted(os.listdir(self.video_path))
        for i in self.video_data:
            self.val_data.append(os.path.join(self.video_path, i))
        self.val_data = sorted(self.val_data)

    def __len__(self):
        return len(self.val_data)
        
    def getimg(self, index):
        data = self.val_data[index]
        img_list = sorted(os.listdir(data)) #一定要sort
        imgs = []
        for i in range(9):
            im = cv2.imread(os.path.join(data, img_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        name = self.video_data[index]
        length = len(imgs)
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)

        return torch.stack(imgs, 0), name#n, c, h , w


class CityTrainDataset(Dataset):
    def __init__(self):
        self.path = './data/cityscapes/train'
        self.train_data = sorted(os.listdir(self.path))

    def __len__(self):
        return len(self.train_data)

    def aug_seq(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs

    def getimg(self, index):
        data_name = self.train_data[index]
        data_path = os.path.join(self.path, data_name)
        frame_list = sorted(os.listdir(data_path))
        imgs = []
        for i in range(30):
            im = cv2.imread(os.path.join(data_path, frame_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        imgs = self.aug_seq(imgs, 256, 256)
        length = len(imgs)
        if random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, ::-1]
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)
        return torch.stack(imgs, 0)#n, c, h , w


class CityValDataset(Dataset):
    def __init__(self):
        self.val_data = []
        self.video_path = './data/cityscapes/test'
        self.video_data = sorted(os.listdir(self.video_path))
        for i in self.video_data:
            self.val_data.append(os.path.join(self.video_path, i))
        self.val_data = sorted(self.val_data)

    def __len__(self):
        return len(self.val_data)
        
    def getimg(self, index):
        data = self.val_data[index]
        img_list = sorted(os.listdir(data))
        imgs = []
        for i in range(14):
            im = cv2.imread(os.path.join(data, img_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        name = self.video_data[index]
        length = len(imgs)
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)

        return torch.stack(imgs, 0), name#n, c, h , w


class MovingMNISTTrainDataset(Dataset):
    urls = [
        'https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    train_file = 'moving_mnist_train.pt'
    test_file = 'moving_mnist_test.pt'

    def __init__(self, args):
        self.root = '/home/mingruibo/data/datasets/movingmnist/MovingMNIST/.data'
        self.split = 1000
        self.download()     # auto download
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it'
        )
        self.train_data = torch.load(
            os.path.join(self.root, self.processed_folder, self.train_file)
        )

    def __len__(self):
        return len(self.train_data)
            
    def __getitem__(self, index):
        data = torch.unsqueeze(self.train_data[index], dim=1) # 10 in, 10 out
        return data.repeat(1, 3, 1, 1)

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.root, self.processed_folder, self.train_file)
        ) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_file)
        )

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip
        if self._check_exists():
            return
        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)
        # process and save as torch files
        print('Processing...')
        training_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[:-self.split]
        )
        test_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[-self.split:]
        )
        with open(os.path.join(self.root, self.processed_folder, self.train_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        print('Done!')


class MovingMNISTValDataset(Dataset):
    urls = [
        'https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    train_file = 'moving_mnist_train.pt'
    test_file = 'moving_mnist_test.pt'

    def __init__(self, args):
        self.root = '/home/mingruibo/data/datasets/movingmnist/MovingMNIST/.data'
        self.split = 1000
        self.download()     # auto download
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it'
        )
        self.val_data = torch.load(
            os.path.join(self.root, self.processed_folder, self.test_file)
        )

    def __len__(self):
        return len(self.val_data)
            
    def __getitem__(self, index):
        data = torch.unsqueeze(self.val_data[index], dim=1) # 10 in, 10 out
        return data.repeat(1, 3, 1, 1), 1

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.root, self.processed_folder, self.train_file)
        ) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_file)
        )

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip
        if self._check_exists():
            return
        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)
        # process and save as torch files
        print('Processing...')
        training_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[:-self.split]
        )
        test_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[-self.split:]
        )
        with open(os.path.join(self.root, self.processed_folder, self.train_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        print('Done!')
