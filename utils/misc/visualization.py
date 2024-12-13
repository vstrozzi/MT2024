from PIL import Image

## Imports
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def _convert_to_rgb(image):
    return image.convert("RGB")


visualization_preprocess = transforms.Compose(
    [
        transforms.Resize(size=224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        _convert_to_rgb,
    ]
)


def image_grid(images, rows, cols, labels=None, scores=None):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img)
        ax.axis('off')
        
        # Display the label and score if provided
        if labels and scores:
            ax.text(0.5, -0.1, f"{labels[i]}: {scores[i]:.5f}", 
                    ha="center", va="top", transform=ax.transAxes, fontsize=10)
    
    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()