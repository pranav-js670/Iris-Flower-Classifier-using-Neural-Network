import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

class ModelWrapper:
    def __init__(self, model_path='Simple Iris Model using Neural Network.pt'):
        self.model = Model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
        with torch.no_grad():
            input_data = torch.tensor([sepal_length, sepal_width, petal_length, petal_width], dtype=torch.float32)
            output_tensor = self.model(input_data.unsqueeze(0))

        _, predicted_index = torch.max(output_tensor, 1)
        if predicted_index == 0:
            return 'Setosa'
        elif predicted_index == 1:
            return 'Versicolor'
        elif predicted_index == 2:
            return 'Virginica'
        
def main():
    st.title('Iris Flower Classification!')
    st.header('Using Neural Network')
    st.subheader('Use the sliders to classify the species of the flower.')
    sepal_length = st.slider('Enter Sepal Length:',4.0, 8.0)
    sepal_width = st.slider('Enter Sepal Width:',2.0, 4.5)
    petal_length = st.slider('Enter Petal Length:',1.0, 7.0)
    petal_width = st.slider('Enter Petal Width:',0.1, 2.5)

    model_wrapper = ModelWrapper()

    if st.button('Predict'):
        prediction = model_wrapper.predict_species(sepal_length, sepal_width, petal_length, petal_width)
        st.success(f'Predicted Species: {prediction}')

if __name__ == '__main__':
    main()

