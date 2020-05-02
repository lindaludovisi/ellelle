from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    
    
    '''
    
    Args:
        root (string): Root directory of dataset 'Caltech101/101_ObjectCategories'
       
        split (string): Specifies if we are working with the train or the test set.
        
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
            
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        
    '''
    
    def __init__(self, root, split='train', transform=None, target_transform=None):
        
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.root = root    #root = 'Caltech101/101_ObjectCategories'
       
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        raw_db = []     #define raw_db : a list containing the paths of the files of our split
        if self.split == 'train':
            my_file = os.path.join(THIS_FOLDER, 'train.txt')
            with open(my_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    raw_db.append(line) #a list of strings: every string is <category>/image_<number>
        elif self.split == 'test':
            my_file = os.path.join(THIS_FOLDER, 'test.txt')
            with open(my_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    raw_db.append(line) #a list of strings: every string is <category>/image_<number>
        else:
            return -1 #error
       
        #remove category "BACKGROUND_Google"
        for elem in raw_db:
            if "BACKGROUND_Google" in elem:
                raw_db.remove(elem) #remove images like "BACKGROUND_Google/image_<number>"
                
        #define self.categories: it is a list containing the names of all the categories, except for BACKGROUND_Google
        self.categories = sorted(os.listdir(self.root)) #order the names of the categories and store them in a list
        self.categories.remove("BACKGROUND_Google") 

        self.index = [] #a list containing all the indexes of the specified split
        self.y = []     #a list containing all the labels. len(self.y)=len(self.index)
        for elem in raw_db:
            words = elem.split('/') #words is a list like [ 'category' , 'image_number' ]
            img = words[1].split('_') #img is a list like [ 'image' , 'number']
            num = img[1]
            self.index.append(int(num[0:4])) #add the number corresponding to the specific image
            
            for i, c in enumerate(self.categories):
                if c == words[0] : 
                    self.y.append(i) #add the number corresponding to the label
            
        
        ''' ORIGINAL VERSION
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, c))) #n=number of images contained in a specific category
            self.index.extend(range(1, n + 1)) #name of the image goes from 1 to n (within the same category)
            self.y.extend(n * [i]) #label is the same for all images belonging to the same category
        '''   
            
    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

         # Provide a way to access image and label via index
         # Image should be a PIL Image
         # label can be int
            
        image = pil_loader(os.path.join(self.root, 
                                        self.categories[self.y[index]],
                                        "image_{:04d}.jpg".format(self.index[index])))
        
     
        label = self.y[index]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.index) # Provide a way to get the length (number of elements) of the dataset
        return length
