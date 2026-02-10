import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import sys
import os

# --- CONFIGURATION ---
MODEL_PATH = "best_mae_model.pth"  # Path to your trained weights
IMAGE_PATH = "test_image.png"      # Put any image path here
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- IMPORTS ---
# Ensure we can find your src folder
if not os.path.exists("src"):
    print("Error: 'src' folder not found. Run this script in the root directory.")
    sys.exit()
    
from src.mae_model import MAE

def prepare_image(image_path):
    """Load, Resize, and Normalize an image for the model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    try:
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0) # Add batch dimension (1, 3, 224, 224)
        return img.to(DEVICE)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit()

def unpatchify(x, patch_size=16):
    """Convert patch embeddings back to whole images."""
    p = patch_size
    h = w = int(x.shape[1]**.5)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def run_inference():
    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = MAE()
    # Load weights (map_location ensures it works on CPU even if trained on GPU)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 2. Load Image
    img = prepare_image(IMAGE_PATH)
    
    # 3. Forward Pass
    with torch.no_grad():
        loss, pred, mask = model(img)
        
        # 4. Reconstruction Logic
        pred_imgs = unpatchify(pred)
        
        # Create mask image (1 = Masked, 0 = Visible)
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, 16**2 * 3)
        mask_imgs = unpatchify(mask) # (1, 3, 224, 224)
        
        # Paper Style Reconstruction:
        # We keep the ORIGINAL visible pixels (sharp) and only fill in the MASKED parts with AI (smooth)
        # Formula: (Original * Visible_Mask) + (Prediction * Hidden_Mask)
        # Note: In our logic, Mask=1 means hidden. So Visible is (1 - Mask).
        reconstruction = (img * (1 - mask_imgs)) + (pred_imgs * mask_imgs)
        
        # Create the "Masked Input" visualization (Gray out the hidden parts)
        masked_input = img * (1 - mask_imgs)

    # 5. Visualization
    print("Visualizing results...")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axs[0].imshow(img[0].cpu().permute(1, 2, 0).clip(0, 1))
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    # Masked
    axs[1].imshow(masked_input[0].cpu().permute(1, 2, 0).clip(0, 1))
    axs[1].set_title("Masked Input (The Challenge)")
    axs[1].axis("off")
    
    # Reconstruction
    axs[2].imshow(reconstruction[0].cpu().permute(1, 2, 0).clip(0, 1))
    axs[2].set_title("MAE Reconstruction")
    axs[2].axis("off")
    
    plt.tight_layout()
    plt.savefig("inference_result.png")
    plt.show()
    print("Done! Saved result to 'inference_result.png'")

if __name__ == "__main__":    
    run_inference()