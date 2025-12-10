import chromadb
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import uuid
import os

class SemanticGatekeeper:
    def __init__(self, db_path="datasets/chroma_db"):
        # Initialize Persistent ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="cl_coco_memory", 
            metadata={"hnsw:space": "cosine"}
        )
        
        # ResNet-18 Feature Extractor (CPU optimized for gating)
        self.device = torch.device("cpu")
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        # Remove classification layer to get 512-dim embeddings
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, image_path):
        """Generates 512-dim embedding from image."""
        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.preprocess(img).unsqueeze(0)
            with torch.no_grad():
                embedding = self.model(img_t).flatten().tolist()
            return embedding
        except Exception as e:
            print(f"Error extracting embedding for {image_path}: {e}")
            return [0.0] * 512

    def is_novel(self, image_path, threshold=0.15):
        """
        Checks if image is semantically unique (Novelty Detection).
        Returns: (bool) True if novel, False if duplicate
        """
        embedding = self.get_embedding(image_path)
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1
        )
        
        # If DB is empty, it is novel
        if not results['distances'] or len(results['distances'][0]) == 0:
            return True
            
        distance = results['distances'][0][0]
        # If distance < threshold (0.15), it is too similar (Duplicate)
        return distance > threshold

    def commit_sample(self, image_path, label, confidence, phase_id):
        """Commits verified sample to academic memory."""
        embedding = self.get_embedding(image_path)
        self.collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            metadatas=[{
                "label": str(label), 
                "confidence": float(confidence),
                "phase": str(phase_id)
            }],
            documents=[image_path]
        )