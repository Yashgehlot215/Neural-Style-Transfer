# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms  # so that we can convert image to tensor
# import torchvision.models as models   #to be able to load VGG19 
# from torchvision.utils import save_image  #to save image

# model = models.vgg19(pretrained=True).features # this will get us all the conv layers

# # print(model) to check which conv layers we want
# class VGG(nn.Module):
#     def __init__(self):
#         super(VGG,self).__init__()
#         self.chosen_features = ['0','5','10','19','28']
#         #the above one are the features we gonna take because that corresponds to 1,1  2,1  3,1  4,1  5,1
#         self.model = models.vgg19(pretrained=True).features[:29]  # because we not gonna use after 29

#     def forward(self,x):
#         features=[]
#         # those are gonna be the relevant features we gonna be storing
#         for layer_num,layer in enumerate(self.model):
#             x = layer(x)
            
#             if str(layer_num) in self.chosen_features:
#                 features.append(x)
#         return features
    
# device = torch.device("cuda" if torch.cuda.is_available else "cpu")
# # function to load image

# def load_image(image_name):
#     image = Image.open(image_name)
#     image = loader(image).unsqueeze(0)
#     return image.to(device)

# image_size = 356
# loader = transforms.Compose(
#     [
#         transforms.Resize((image_size,image_size)),
#         transforms.ToTensor(),
#         #transforms.Normalize(mean=[],std=[]) It is used to improve result a little bit

#     ]
# )

# original_img = load_image("golden_gate2.jpg")
# style_img = load_image("style4.png")
# mode = VGG().to(device).eval()

# # generated = torch.randn(original_img.shape,device=device,requires_grad=True)
# generated = original_img.clone().requires_grad_(True)

# #Hyper paramters
# total_steps = 5000
# learning_rate = 0.001
# alpha = 1
# beta = 0.01

# optimizer = optim.Adam([generated],lr=learning_rate)
# for step in range(total_steps):
#     generated_features = model(generated)
#     original_img_features = mode(original_img)
#     style_features = mode(style_img)

#     style_loss = original_loss =0  #original loss is basically content loss

#     for gen_feature,orig_feature,style_feature in zip(
#         generated_features,original_img_features,style_features
#     ):
#         batch_size ,channel,height,width = gen_feature.shape
#         original_loss+=torch.mean((gen_feature-orig_feature)**2)

#         # computer gram_matrix for gen and style

#         G = gen_feature.view(channel,height*width).mm(
#             gen_feature.view(channel,height*width).t()
#         )
#         A = style_feature.view(channel,height*width).mm(
#             style_feature.view(channel,height*width).t()
#         )
#         style_loss +=torch.mean((G-A)**2)
        
#     total_loss = alpha*original_loss+ beta*style_loss
#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()
#     if step%200==0:
#         print(total_loss)
#         save_image(generated,"generated.png")


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# import torchvision.models as models
# from torchvision.utils import save_image

# # Ensure the correct model weights parameter
# model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

# class VGG(nn.Module):
#     def __init__(self):
#         super(VGG, self).__init__()
#         self.chosen_features = ['0', '5', '10', '19', '28']
#         self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:29]

#     def forward(self, x):
#         features = []
#         for layer_num, layer in enumerate(self.model):
#             x = layer(x)
#             if str(layer_num) in self.chosen_features:
#                 features.append(x)
#         return features

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_image(image_name):
#     image = Image.open(image_name)
#     image = loader(image).unsqueeze(0)
#     return image.to(device)

# image_size = 356
# loader = transforms.Compose([
#     transforms.Resize((image_size, image_size)),
#     transforms.ToTensor(),
# ])

# # Paths to your images
# original_img_path = "D:\\Aries open project\\PART2\\golden_gate2.jpg"
# style_img_path = "D:\\Aries open project\\PART2\\candy.jpg"

# original_img = load_image(original_img_path)
# style_img = load_image(style_img_path)
# model = VGG().to(device).eval()

# generated = original_img.clone().requires_grad_(True)

# # Hyperparameters
# total_steps = 5000
# learning_rate = 0.001
# alpha = 1
# beta = 0.01

# optimizer = optim.Adam([generated], lr=learning_rate)
# for step in range(total_steps):
#     generated_features = model(generated)
#     original_img_features = model(original_img)
#     style_features = model(style_img)

#     style_loss = original_loss = 0

#     for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_features):
#         batch_size, channel, height, width = gen_feature.shape
#         original_loss += torch.mean((gen_feature - orig_feature) ** 2)

#         # Compute the Gram matrix for gen and style
#         G = gen_feature.view(channel, height * width).mm(gen_feature.view(channel, height * width).t())
#         A = style_feature.view(channel, height * width).mm(style_feature.view(channel, height * width).t())
#         style_loss += torch.mean((G - A) ** 2)

#     total_loss = alpha * original_loss + beta * style_loss
#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()

#     if step % 200 == 0:
#         print(f"Step [{step}/{total_steps}], Total Loss: {total_loss.item()}")
#         save_image(generated, f"generated_{step}.png")

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

# Ensure the correct model weights parameter
model = models.vgg19(pretrained=True).features

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_name):
    image_size = 356  # Define image size here
    loader = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Paths to your images
original_img_path = "D:\\Aries open project\\PART2\\lion.jpg"
style_img_path = "D:\\Aries open project\\PART2\\candy.jpg"

original_img = load_image(original_img_path)
style_img = load_image(style_img_path)

# Initialize VGG and move to device
model = VGG().to(device).eval()

# Initialize generated image as a copy of original image
generated = original_img.clone().requires_grad_(True)

# Hyperparameters
total_steps = 5000
learning_rate = 0.01  # Adjusted learning rate for faster convergence
alpha = 1  # Weight for content loss
beta = 0.1  # Weight for style loss

optimizer = optim.Adam([generated], lr=learning_rate)

for step in range(total_steps):
    optimizer.zero_grad()

    # Extract features for generated, original, and style images
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    style_loss = original_loss = 0

    # Compute content loss
    for gen_feature, orig_feature in zip(generated_features, original_img_features):
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)

    # Compute style loss
    for gen_feature, style_feature in zip(generated_features, style_features):
        _, channel, height, width = gen_feature.shape
        G = gen_feature.view(channel, height * width).mm(gen_feature.view(channel, height * width).t())
        A = style_feature.view(channel, height * width).mm(style_feature.view(channel, height * width).t())
        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss
    total_loss.backward()
    optimizer.step()

    # Print and save generated image every 200 steps
    if step % 200 == 0:
        print(f"Step [{step}/{total_steps}], Total Loss: {total_loss.item()}")
        save_image(generated, f"generated_{step}.png")














