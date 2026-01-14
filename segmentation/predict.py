import torch
import cv2
import os
from training.unet import UNet

# Lazy model loading
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model():
    global _model
    if _model is None:
        model_path = "model/tongue_unet.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first by running: python training/train.py")
        try:
            _model = UNet()
            _model.load_state_dict(torch.load(model_path, map_location=_device))
            _model.to(_device)
            _model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    return _model

def segment_tongue(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to read image: {input_path}")
    
    try:
        image = cv2.resize(image, (768, 576))
        tensor = torch.tensor(image / 255.0).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(_device)
        
        model = _load_model()
        
        with torch.no_grad():
            output = model(tensor)
            mask = output[0][0].cpu().numpy()

        mask = (mask > 0.5).astype("uint8") * 255
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        segmented = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite(output_path, segmented)
        
        if not os.path.exists(output_path):
            raise RuntimeError(f"Failed to save segmented image to {output_path}")
            
    except Exception as e:
        raise RuntimeError(f"Segmentation failed: {str(e)}")
