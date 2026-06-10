import json
import os
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

DATA_DIR = Path(__file__).parent
DB_FILE = DATA_DIR / "polish_landmarks_db.json"
REF_IMAGES_DIR = DATA_DIR / "reference_images"
VECTORS_FILE = DATA_DIR / "landmark_vectors.pt"

# Cache dir as defined in geolocator.py
CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "/tmp/models")

def build_vector_db():
    if not DB_FILE.exists():
        print(f"Error: Database file {DB_FILE} not found.")
        return
        
    print("Loading landmark metadata...")
    with open(DB_FILE, "r", encoding="utf-8") as f:
        landmarks = json.load(f)
        
    valid_landmarks = [lm for lm in landmarks if lm.get("image_filename")]
    print(f"Found {len(valid_landmarks)} landmarks with images.")
    
    if len(valid_landmarks) == 0:
        print("No images to process.")
        return

    # Check for torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading StreetCLIP model...")
    repo_id = "geolocal/StreetCLIP"
    processor = CLIPProcessor.from_pretrained(repo_id, cache_dir=CACHE_DIR)
    model = CLIPModel.from_pretrained(repo_id, cache_dir=CACHE_DIR)
    model.eval()
    model.to(device)
    
    print("Generating image embeddings...")
    vector_db = {}
    
    with torch.no_grad():
        for lm in tqdm(valid_landmarks):
            name = lm["name"]
            clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
            lm_dir = REF_IMAGES_DIR / clean_name
            
            # Legacy fallback: pojedynczy plik z `image_filename`
            legacy_file = REF_IMAGES_DIR / lm.get("image_filename", "")
            
            image_paths = []
            if lm_dir.exists() and lm_dir.is_dir():
                image_paths.extend(list(lm_dir.glob("*.jpg")) + list(lm_dir.glob("*.jpeg")) + list(lm_dir.glob("*.png")))
            elif legacy_file.exists() and legacy_file.is_file():
                image_paths.append(legacy_file)
                
            if not image_paths:
                print(f"Warning: No images found for {name}")
                continue
                
            lm_vectors = []
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    image_features = model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    lm_vectors.append(image_features.cpu().squeeze(0))
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
            
            if lm_vectors:
                # Store matrix of shape (N, 512) for N images
                vector_db[name] = torch.stack(lm_vectors)
                
    print(f"Generated embeddings for {len(vector_db)} landmarks.")
    
    # Save the database
    torch.save(vector_db, VECTORS_FILE)
    print(f"Saved vector database to {VECTORS_FILE}")

if __name__ == "__main__":
    build_vector_db()
