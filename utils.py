import os
import datasets
from transformers import ViTFeatureExtractor
from tensorflow import keras

def create_image_folder_dataset(root_path):
  
  features=datasets.Features({
                      "img": datasets.Image(),
                      "filename": datasets.features.Value(dtype='string', id=None),
                  })

  img_data_files=[]
  filenames = []

  for img in os.listdir(os.path.join(root_path)):
    path_=os.path.join(root_path,img)
    filenames.append(img)
    img_data_files.append(path_)

  ds = datasets.Dataset.from_dict({"img":img_data_files, "filename":filenames},features=features)
  return ds


feature_extractor = ViTFeatureExtractor(do_resize = True, size = 224, do_normalize = True)

def process(examples):
    try:
      examples.update(feature_extractor(examples['img'], ))
    except:
      print(examples)
    return examples
