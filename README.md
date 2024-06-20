# Neural-Style-Transfer

## Introduction 
This projects implements neural style transfer using the VGG19 Model in PyTorch. NST is a technique for combining the content of one image with the style of another image to create new, visually appealing image.

For Project Report,[Click here](https://docs.google.com/document/d/1aMs9dBNpfnUQYe7WOupe4KDcikSqZIGcI6BF2hNMbnE/edit?usp=sharing).

## Table of Contents :bar_chart:
- Requirements
- Datasets Used
- Data Preprocessing
- Models Architecture
- Model Training and Hyperparameters
- Model Evaluation
- How To Run
- Improvements
- References

## Requirements: 🔔
``` 
torch
torchvision
Pillow
```
We can install these dependencies using pip:
```bash
pip install torch torchvision Pillow
```

## Data Used :school_satchel:
For Pre Processed ImageNet Data Set was used on VGG19 Model to train that.
Apart from that here the project utilizes two images: content image and a style image. These images are used to blend the content and style to create the output image.


## Data PreProcessing
Images are loaded and preprocessed using the following steps:
- Resized to 356*356 pixels
- Converted to Tensors
- Normalized for the VGG19 model
``` python
def load_image(image_name):
    image_size = 356
    loader = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
```

## Model Architecture 
The VGG19 Model is used for feature extraction, focusing on the first 29 layers to capture both content and style features. This model is modified to return features fro specific layers.
```python

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
```

## Training and Hyperparameters:🎯
The training loop iterativelty updates the generated image to minimize the combined content and style loss
- Patch size: 356 x 356
- Batch size: 1
- Optimizer: Adam
- Learning rate: 0.01
- Epochs: 5000
- Content Weight (alpha): 1
- Style Weight (beta): 0.1
### Training Loop
``` python
for step in range(total_steps):
    optimizer.zero_grad()
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)
    
    # Compute losses
    original_loss = 0
    for gen_feature, orig_feature in zip(generated_features, original_img_features):
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)
    
    style_loss = 0
    for gen_feature, style_feature in zip(generated_features, style_features):
        _, channel, height, width = gen_feature.shape
        G = gram_matrix(gen_feature)
        A = gram_matrix(style_feature)
        style_loss += torch.mean((G - A) ** 2)
    
    total_loss = alpha * original_loss + beta * style_loss
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step [{step}/{total_steps}], Total Loss: {total_loss.item()}")
        save_image(generated, f"images/generated_{step}.png")
```

## Results and Analysis
Generated images are saved periodically during training to monitor the progression of style transfer.Total loss values are printed to evaluate convergence and the quality of generated images.

Results of content and style image used in demo videos are saved in folder **output_images** folders. And other folders contains content and style images on which we can test our model.

## How To Run:

Clone the Repository:
``` bash
git clone https://github.com/Yashgehlot215/Neural-Style-Transfer
```
Install the required dependencies:
``` bash
cd Neural-Style-Transfer
pip install -r requirements.txt
```

Run the Jupyter Notebook:
``` bash
jupyter notebook Neural_Style_Transfer.ipynb
```

Ensure that content and style images are placed in correct directory and locations in code are updated before testing the model.

## Improvements:
- **Early Stopping** :Implement early stopping to prevent overfitting by monitoring validation loss and stopping training when it starts to increase.
- **Advanced Optimizers**:Experiment with advanced optimizers like RMSprop or AdamW to potentially improve convergence speed and stability.

## Reference:📎:
1. https://pytorch.org/docs/stable/index.html
2. https://pytorch.org/vision/stable/models.html
3. https://arxiv.org/abs/1508.06576
