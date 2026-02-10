import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import csv
from tqdm.notebook import tqdm

BATCH_SIZE = 64
LR = 1.5e-4
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if not os.path.exists("./imagenette2-320"):
        print("Downloading ImageNette...")
        os.system("wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz")
        os.system("tar -xf imagenette2-320.tgz")
    
    train_dataset = datasets.ImageFolder(root="./imagenette2-320/train", transform=transform)
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

def unpatchify(x, patch_size=16):
    p = patch_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def visualize_results(model, epoch=None):
    model.eval()
    dataloader = get_dataloader()
    imgs, _ = next(iter(dataloader))
    imgs = imgs[:4].to(DEVICE)
    
    with torch.no_grad():
        loss, pred, mask = model(imgs)
        
        pred_imgs = unpatchify(pred)
        
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, 16**2 *3)
        mask_imgs = unpatchify(mask)
        
        im_masked = imgs * (1 - mask_imgs)
        im_paste = (imgs * (1 - mask_imgs)) + (pred_imgs * mask_imgs)

    fig, axs = plt.subplots(4, 3, figsize=(12, 16))
    for i in range(4):
        axs[i, 0].imshow(imgs[i].cpu().permute(1, 2, 0).clip(0, 1))
        axs[i, 0].set_title("Original")
        axs[i, 0].axis("off")
        
        axs[i, 1].imshow(im_masked[i].cpu().permute(1, 2, 0).clip(0, 1))
        axs[i, 1].set_title("Masked Input")
        axs[i, 1].axis("off")
        
        axs[i, 2].imshow(im_paste[i].cpu().permute(1, 2, 0).clip(0, 1))
        axs[i, 2].set_title("Reconstruction")
        axs[i, 2].axis("off")
    
    plt.tight_layout()
    filename = f"mae_reconstruction_epoch_{epoch}.png" if epoch else "mae_reconstruction_final.png"
    plt.savefig(filename)
    plt.show()
    plt.close()

def train():
    import sys
    sys.path.append("/kaggle/input/datasets/abdelrhmaneldenary/mae-source-code/src")
    from mae import MAE 

    print(f"Setting up training on {DEVICE}...")
    dataloader = get_dataloader()
    model = MAE().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    
    best_loss = float('inf')
    loss_history = []
    
    # Early Stopping Variables
    patience = 5
    patience_counter = 0
    
    log_file = open("training_log.csv", "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["Epoch", "Average_Loss"])

    print(f"Starting training for {EPOCHS} epochs...")

    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for batch_idx, (imgs, _) in enumerate(progress_bar):
                imgs = imgs.to(DEVICE)
                
                loss, pred, mask = model(imgs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(dataloader)
            loss_history.append(avg_loss)
            
            print(f"Epoch {epoch+1} Completed. Average Loss: {avg_loss:.6f}")
            
            logger.writerow([epoch+1, avg_loss])
            log_file.flush()
            
            # Save Best Model Logic
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0  # Reset counter if model improved
                torch.save(model.state_dict(), "best_mae_model.pth")
                print(f"--> Best Model Saved! (Loss: {avg_loss:.6f})")
            else:
                # Early Stopping Logic (Only checks after epoch 20)
                if epoch >= 20:
                    patience_counter += 1
                    print(f"No improvement. Early Stopping Counter: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print("Early Stopping Triggered: Model stopped learning.")
                        break

            if (epoch + 1) % 5 == 0:
                print(f"Visualizing results for Epoch {epoch+1}...")
                visualize_results(model, epoch=epoch+1)
                model.train() 

    except KeyboardInterrupt:
        print("Training interrupted manually. Saving current progress...")

    finally:
        log_file.close()
        torch.save(model.state_dict(), "final_mae_model.pth")
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(loss_history)+1), loss_history, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('MAE Training Progress')
        plt.legend()
        plt.grid(True)
        
        # Force X-Axis to be Integers only
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.savefig("training_loss_curve.png")
        plt.show()
        
        return model

if __name__ == "__main__":
    trained_model = train()