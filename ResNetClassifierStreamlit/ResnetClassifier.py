import streamlit as st
import io
from PIL import Image
import torch
from torchvision.models import resnet50
from torchvision import transforms
import plotly.graph_objects as go
import utils
import requests

# Load the model only once
resnet50 = resnet50(pretrained=True)
resnet50.eval()

def load_class_names(url):
    response = requests.get(url)
    class_names = response.text.split('\n')
    return class_names

imagenet_classes_url = "https://raw.githubusercontent.com/OscarAhumadaG/ResNetClassifier/main/ResNetClassifierStreamlit/imagenet-classes.txt"

class_names = load_class_names(imagenet_classes_url)
st.write(class_names)
st.title("ResNet CNN Classifier")

uploaded_file = st.sidebar.file_uploader("Choose an image")
btn_classify = st.sidebar.button("Classify")

# Check if a file is uploaded
if btn_classify and  uploaded_file is not None:
    # Use try-except to handle potential errors when opening the image
    try:
        # Open the image using PIL
		
        image = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Perform classification or other tasks here if needed
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        
        input_tensor = transform(image)
    
        input_batch = input_tensor.unsqueeze(0)
    
        with torch.no_grad():
            output = torch.nn.functional.softmax(resnet50(input_batch), dim=1)
    
        results = utils.print_prob(output[0], class_names, top_n=1)
	st.write(results)

	    
        """st.title("Image Results")
        
        for idx, result in enumerate(results):
            labels = []
            scores = []
    # Iterate through the list of predictions in each result
            for prediction in result:
                # Extract the label and score from each prediction
                if len(prediction) == 2:
                    label, score = prediction
                    label = label.split(',')[0]  # Extract only the first part of the label
                    score = float(score.rstrip('%'))  # Convert the score to a float
                    labels.append(label)
                    scores.append(score)

                    # Display the formatted result
                    st.write(f"{label.title()}: {score:.2f}%")
                else:
                    st.error("Invalid prediction format. Expected (label, score).")"""
        
        st.write()
        st.title("Visualization Results")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=scores, name='Prediction Scores', text=scores, textposition='auto'))
        fig.update_layout(title_text='Prediction Scores', xaxis_title='Labels', yaxis_title='Scores (%)')
        # Display the Plotly figure using st.plotly_chart
        st.plotly_chart(fig)
        
        

    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")




