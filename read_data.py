from torch.utils.data import  Dataset
from PIL import  Image
import os
class MyData(Dataset):
    #self这里可以说是相当于创建全局变量，可以给后面的函数使用。
    #Mydata类的全局变量有 root_dir,label_dir,path,img_path
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dir = "learn_torch/dataset/train"
ants_label_dir = "ants"
bee_label_dir = "bees"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bee_label_dir)

train_dataset = ants_dataset + bees_dataset
