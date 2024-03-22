import streamlit as st
import io
from PIL import Image
import torch
from torchvision.models import resnet50
from torchvision import transforms
import plotly.graph_objects as go
import requests

# Load the model only once
resnet50_model = resnet50(pretrained=True)
resnet50_model.eval()

def load_class_names(url):
    response = requests.get(url)
    class_names = response.text.split('\n')
    return class_names

imagenet_classes_url = "https://raw.githubusercontent.com/OscarAhumadaG/ResNetClassifier/main/ResNetClassifierStreamlit/imagenet-classes.txt"
class_names = load_class_names(imagenet_classes_url)

st.title("ResNet CNN Classifier")

uploaded_file = st.sidebar.file_uploader("Choose an image")
btn_classify = st.sidebar.button("Classify")

# Check if a file is uploaded
if btn_classify and uploaded_file is not None:
    # Use try-except to handle potential errors when opening the image
    try:
        # Open the image using PIL
        image = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Perform classification
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = torch.nn.functional.softmax(resnet50_model(input_batch), dim=1)
        
        # Get the top 5 predicted classes and scores
        top5_scores, top5_indices = torch.topk(output, 5)
        top5_scores = top5_scores.squeeze().tolist()
        top5_indices = top5_indices.squeeze().tolist()
        
        # Get the labels corresponding to the top 5 indices
        top5_labels = [class_names[idx] for idx in top5_indices]
        
        # Display the top 5 predicted classes and scores
        st.write("Top 5 Predictions:")
        for label, score in zip(top5_labels, top5_scores):
            st.write(f"{label}: {score:.4f}")
        
        # Visualization
        st.title("Visualization Results")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top5_labels, y=top5_scores, name='Prediction Scores'))
        fig.update_layout(title_text='Top 5 Prediction Scores', xaxis_title='Labels', yaxis_title='Scores')
        # Display the Plotly figure using st.plotly_chart
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
