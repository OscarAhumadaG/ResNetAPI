from fastapi import FastAPI, HTTPException, Path, Query, UploadFile, File
import base64
from PIL import Image
import json
import requests
import io
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import uvicorn

# Kaveh, I used the port 8002 because in my laptop the 8000 was in used


app = FastAPI(title="ResNet Classifier")



@app.get("/")
def read_root():
    return {"Hello": "World"}




@app.get("/classify")
def Resnet_ClassificationGet():
    return {"Hello": "World"} 


@app.post("/classify")
async def Resnet_Classification(file: UploadFile):
    resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

    resnet50.eval()

    request_object_content = await file.read()
    image = Image.open(io.BytesIO(request_object_content)).convert('RGB')
    
    #with open("Laptop.jpg", 'rb') as image_file:
        #image_bytes = io.BytesIO(image_file.read())
        #image_bytes.seek(0)  # Reset the position to the beginning
        #image = Image.open(image_bytes).convert('RGB')
        
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        
    input_tensor = transform(image)
    
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = torch.nn.functional.softmax(resnet50(input_batch), dim=1)
    
    results = utils.pick_n_best(predictions=output, n=4)


    
    return {"prediction": results}
        

if __name__ == "__main__":
    uvicorn.run("FastAPI_ResnetClassification:app", 
                host="127.0.0.1", 
                port=8002, 
                reload=True)
