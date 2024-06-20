
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














