import sys
import json
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(directory):
    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    main_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
    ])
    }
    class_names = ['female', 'male']
    result = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            im = Image.open(image_path)
            image_tensor = main_transforms['test'](im).float()
            image_tensor = image_tensor.unsqueeze_(0)
            input = Variable(image_tensor)
            input = input.to(device)
            output = main_model(input)
            index = output.data.cpu().numpy().argmax()
            result[image_path] = class_names[index + 1]
    with open("process_results.json", 'w') as f:
        f.write(json.dumps(result))

if __name__ == "__main__":
    main_model = torch.load(**PATH_TO_MODEL**) #подставить путь к модели
    main_model.eval()
    predict(sys.argv[1])
