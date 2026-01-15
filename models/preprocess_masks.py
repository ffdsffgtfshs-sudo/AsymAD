import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from typing import List

def load_birefnet_model(device: torch.device):
    """
    Load the BiRefNet-HRSOD model using a specific local cache directory.
    """
    # Get current working directory
    current_dir = os.getcwd()
    # Define target cache path: ./weights/BiRefNet
    custom_cache_dir = os.path.join(current_dir, "weights", "BiRefNet")
    os.makedirs(custom_cache_dir, exist_ok=True)

    model_id = 'ZhengPeng7/BiRefNet-HRSOD'
    print(f"[Info] Model cache path: {custom_cache_dir}")
    print(f"[Info] Loading BiRefNet-HRSOD model: {model_id} ...")

    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=custom_cache_dir
        )
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return None

def process_single_image_birefnet(model, image_path: str, output_path: str, device: torch.device, transform_image):
    """
    Process a single image using the BiRefNet model and save the binary mask.
    """
    try:
        # Load and convert image
        image = Image.open(image_path).convert('RGB')
        w, h = image.size

        # Preprocessing
        input_tensor = transform_image(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            preds = model(input_tensor)
            if isinstance(preds, (list, tuple)):
                pred_logits = preds[-1]
            else:
                pred_logits = preds

        # Post-processing
        pred_map = torch.sigmoid(pred_logits)
        # Resize back to original dimensions
        pred_map = F.interpolate(pred_map, size=(h, w), mode='bilinear', align_corners=False)
        pred_map = pred_map.squeeze().cpu()

        # Binarization (threshold 0.5)
        binary_mask = (pred_map > 0.5).float()

        # Save as image (Mode 'L' for grayscale)
        mask_image = transforms.ToPILImage()(binary_mask)
        mask_image.save(output_path)
        
    except Exception as e:
        print(f"[Error] Failed to process image {image_path}: {e}")

def process_single_image_white(image_path: str, output_path: str):
    """
    Generate a full white mask based on the original image size and save it.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        
        # Create a full white image (Mode 'L', value 255)
        white_mask = Image.new('L', (w, h), 255)
        white_mask.save(output_path)
        
    except Exception as e:
        print(f"[Error] Failed to generate white mask for {image_path}: {e}")

def preprocess_datasets(data_root: str, obj_list: List[str], texture_list: List[str]):
    """
    Main interface to generate masks for datasets.
    
    Args:
        data_root (str): Root directory of the dataset.
        obj_list (List[str]): List of object categories (uses BiRefNet model).
        texture_list (List[str]): List of texture categories (uses full white masks).
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Device used: {device}")

    # --- Step 1: Process Object Categories (BiRefNet) ---
    print("-" * 50)
    print("[Step 1] Preparing BiRefNet model for object categories...")
    
    model = load_birefnet_model(device)
    
    if model is None:
        print("[Error] Model loading failed. Skipping List 1 processing.")
    else:
        # Define Transforms (HRSOD suggested size: 1024)
        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for item in obj_list:
            source_dir = os.path.join(data_root, item, 'train', 'good')
            target_dir = os.path.join(data_root, item, 'train', 'mask')

            if not os.path.exists(source_dir):
                print(f"[Warning] Path not found: {source_dir}, skipping.")
                continue
            
            os.makedirs(target_dir, exist_ok=True)
            
            files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            print(f"Processing Category [Model]: {item} ({len(files)} images)")

            for file_name in tqdm(files, desc=f"Processing {item}"):
                input_path = os.path.join(source_dir, file_name)
                output_path = os.path.join(target_dir, file_name)
                
                # Overwrite by default
                process_single_image_birefnet(model, input_path, output_path, device, transform_image)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # --- Step 2: Process Texture Categories (White Mask) ---
    print("\n" + "-" * 50)
    print("[Step 2] Processing texture categories (White Masks)...")

    for item in texture_list:
        source_dir = os.path.join(data_root, item, 'train', 'good')
        target_dir = os.path.join(data_root, item, 'train', 'mask')

        if not os.path.exists(source_dir):
            print(f"[Warning] Path not found: {source_dir}, skipping.")
            continue

        os.makedirs(target_dir, exist_ok=True)

        files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"Processing Category [White]: {item} ({len(files)} images)")

        for file_name in tqdm(files, desc=f"Processing {item}"):
            input_path = os.path.join(source_dir, file_name)
            output_path = os.path.join(target_dir, file_name)
            
            process_single_image_white(input_path, output_path)

    print("\n[Finished] All preprocessing tasks completed.")