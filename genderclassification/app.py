import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


file_up = st.file_uploader("Upload an image", type="jpg")

if file_up:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    resnet = models.resnet34(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])

    def predict(model, input):
        import urllib.request
        
        checkpoint = torch.load('point_resnet34_8020_best.pth', map_location=torch.device('cpu'))
        # checkpoint = torch.hub.load_state_dict_from_url('https://drive.google.com/uc?export=download&id=1-2IElarIVhAQ9dvj6ZWxt9MAEBKluoTl', map_location=torch.device('cpu'))
        class_mapping = ["female", "male"]
        model.load_state_dict(checkpoint['model'])
        model.eval()
        with torch.no_grad():
            return class_mapping[model(input)[0].argmax(0)]

    img = torch.unsqueeze(preprocess(image), 0)
    out = predict(resnet, img)
    st.info(out)