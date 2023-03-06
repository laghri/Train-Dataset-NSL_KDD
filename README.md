# Train-Dataset-NSL_KDD
This project involves training a deep learning model on the NSL_KDD dataset for Intrusion Detection Systems (IDS). The trained model can help identify potential intrusions and enhance the security of a system.

## Table of Contents
- Technologies Used
-Installation
-Usage
-Dataset
-Results
-Contributing
-License
-Installation

# Technologies Used
This project uses several technologies, including:
- TensorFlow
- Pandas
- NumPy
- Seaborn
- Matplotlib
## Results
The trained mode
### To use this project, first clone the repository:
 git clone https://github.com/laghri/Train-Dataset-NSL_KDD.git
### Then, install the required dependencies:
pip install tensorflow,pandas,Numpy...
## Usage
To use the trained model, run the Flask web application:
- python app.py
You can then access the web interface at http://localhost:5000. Follow the instructions on the web page to classify network traffic and detect intrusions.

## Dataset
The NSL_KDD dataset is a widely-used benchmark dataset for IDS. It consists of network traffic data and associated labels indicating whether the traffic is normal or anomalous. In this project, the dataset was preprocessed to extract features and normalize the data.

## Model Architecture
The deep learning model used in this project is a convolutional neural network (CNN) based on the ResNet architecture. The model was trained using TensorFlow and achieved high accuracy on the NSL_KDD dataset.

## Results
The trained model achieved an accuracy of 98% on the NSL_KDD dataset. The web interface allows users to classify network traffic and detect potential intrusions in real-time.

.
