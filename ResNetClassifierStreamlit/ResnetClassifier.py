import streamlit as st
import io
from PIL import Image
import torch
from torchvision.models import resnet50
from torchvision import transforms
import plotly.graph_objects as go
import utils  # Assuming you have a utils.py file with pick_n_best function

# Load the model only once
resnet50_model = resnet50(pretrained=True)
resnet50_model.eval()

st.title("ResNet CNN Classifier")

uploaded_file = st.sidebar.file_uploader("Choose an image")
btn_classify = st.sidebar.button("Classify")

if uploaded_file is not None:
    st.write("Image uploaded!")
    image = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

if btn_classify and uploaded_file is not None:
    st.write("Processing the image...")

    try:
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

        results = utils.pick_n_best(predictions=output, n=4)

        st.title("Image Results")

        labels = []
        scores = []

        for idx, result in enumerate(results):
            for prediction in result:
                if len(prediction) == 2:
                    label, score = prediction
                    label = label.split(',')[0]
                    score = float(score.rstrip('%'))
                    labels.append(label)
                    scores.append(score)
                    st.write(f"{label.title()}: {score:.2f}%")
                else:
                    st.error("Invalid prediction format. Expected (label, score).")

        st.write()
        st.title("Visualization Results")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=scores, name='Prediction Scores', text=scores, textposition='auto'))
        fig.update_layout(title_text='Prediction Scores', xaxis_title='Labels', yaxis_title='Scores (%)')
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
elif btn_classify:
    st.warning("Please upload an image first.")
