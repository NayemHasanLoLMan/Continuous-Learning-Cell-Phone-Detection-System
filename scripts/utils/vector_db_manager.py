import chromadb
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import uuid
import os

class VectorDBManager:
    def __init__(self, db_path="datasets/chroma_db", collection_name="cellphone_memory"):
        # Initialize ChromaDB (Persistent)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity for images
        )

        # Initialize ResNet18 for Feature Extraction (CPU is fine for inference)
        self.device = torch.device("cpu") 
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Strip the classification layer to get raw embeddings (512 dimensions)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, image_input):
        """Generates a 512-dim vector from a file path or PIL Image."""
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input.convert('RGB')

        img_tensor = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(img_tensor)
        
        return embedding.flatten().numpy().tolist()

    def is_semantically_unique(self, image_input, threshold=0.15):
        """
        Returns True if the image is NOVEL (unique).
        threshold: 0.0 (identical) to 1.0 (opposite). 
        0.15 is a good starting point for 'visually similar'.
        """
        embedding = self.get_embedding(image_input)
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1
        )

        # If DB is empty, it is unique
        if not results['distances'] or len(results['distances'][0]) == 0:
            return True, embedding

        similarity_distance = results['distances'][0][0]
        
        # If distance is LOW, it is a DUPLICATE.
        # We return True (Unique) only if distance > threshold
        is_unique = similarity_distance > threshold
        return is_unique, embedding

    def add_to_memory(self, image_path, label, confidence, embedding=None):
        """Adds verified data to the vector DB."""
        if embedding is None:
            embedding = self.get_embedding(image_path)
            
        self.collection.add(
            documents=[image_path],
            metadatas=[{"label": label, "confidence": float(confidence)}],
            ids=[str(uuid.uuid4())],
            embeddings=[embedding]
        )