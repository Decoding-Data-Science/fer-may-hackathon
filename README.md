# FER May Hakathon

Facial Emotion Detection Hackathon Project, Create a model and test it using 5 to 10 second videos to detect emotions 

**Please watch the demonstration video along with testing the app** as in the video, I discuss an *alternate approach* to doing this which I was not able to implement in time. In the video, I used clips from the CREMA-D dataset. CREMA-D is a data set of 7,442 original clips from 91 actors. These clips were from 48 male and 43 female actors between the ages of 20 and 74 coming from a variety of races and ethnicities (African America, Asian, Caucasian, Hispanic, and Unspecified). Actors spoke from a selection of 12 sentences. The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad) and four different emotion levels (Low, Medium, High, and Unspecified).

### Team Name: MLX
### Member: Suhail Ahmed | suhailz13ahmed@outlook.com
### Deployed Link: https://fer-dds-mlx.streamlit.app

### Model Accuracy: 35.77% on the validation dataset - FER2013 Partial (as provided in the repository).

## Implemented Methodology:

### Machine Learning Model

I implemented a CNN model and trained it on the FER 2013 partial daataset for 75 epochs. 

The model is built using the Keras Sequential API. It consists of multiple layers of convolutional neural networks, batch normalization, max-pooling, and dropout layers to prevent overfitting. The architecture is designed to progressively extract higher-level features from the input images.

- **Input Layer:** The input layer expects images of shape (48, 48, 1), which corresponds to grayscale images of size 48x48 pixels.

- **Convolutional Layers:** These layers use 3x3 filters to convolve the input and extract features. The activation function used is ReLU (Rectified Linear Unit).

- **Batch Normalization:** This layer normalizes the outputs of the previous layer to stabilize and accelerate the training process.

- **Max-Pooling Layers:** These layers downsample the input by taking the maximum value in each 2x2 pool, reducing the spatial dimensions.

- **Dropout Layers:** These layers randomly drop a fraction of the units during training to prevent overfitting.

- **Flatten Layer:** This layer flattens the 3D output from the convolutional layers into a 1D vector, which is fed into the dense (fully connected) layers.

- **Dense Layers:** These layers perform the final classification. The last dense layer uses a softmax activation function to output probabilities for each of the seven emotion classes.

#### Compilation
The model is compiled using the Adam optimizer with the specified learning rate. The loss function used is categorical cross-entropy, which is suitable for multi-class classification problems. Accuracy is used as the evaluation metric.

##### Callbacks
Three callbacks are used during training to improve performance and prevent overfitting:

ReduceLROnPlateau: This callback reduces the learning rate when the validation loss plateaus, helping the model converge.
EarlyStopping: This callback stops training when the validation accuracy does not improve for a specified number of epochs, preventing overfitting.
ModelCheckpoint: This callback saves the model weights when the validation loss improves, ensuring the best model is saved.

#### Model Training
The model is trained using the fit method. The training data is split into training and validation sets, and the model is trained for the specified number of epochs and batch size. The shuffle parameter ensures that the data is shuffled before each epoch.

#### Rationale

##### Convolutional Neural Network
The use of a convolutional neural network (CNN) is appropriate for image classification tasks due to its ability to automatically learn spatial hierarchies of features from input images. The multiple convolutional layers with increasing filter sizes help the model capture complex patterns in the data.

##### Regularization
Dropout and batch normalization are used extensively throughout the network to prevent overfitting and improve generalization. The l2 regularization on the first layer also helps in reducing overfitting by penalizing large weights.

##### Optimizer
The Adam optimizer is chosen for its efficiency and adaptive learning rate, which helps in faster convergence compared to traditional stochastic gradient descent.


### Streamlit Application

Please nagivate to the dds directory and once you are in it, run:

```console
streamlit run app.py
```

Upon uploading you video, it will take some time to process it, depending on the number of pixels and length of the video. After its processing, you will be able to see all the detected emotions per frame and the most detected emotion is the predicted emotion.

<br>

<img width="766" alt="image" src="https://github.com/Suhail270/fer-may-hackathon/assets/57321434/fa700a38-f84c-441a-8a6c-5ef44daeff36">

<br>
<br>
You may also scroll below and use the slider to view the emotion detected at a particular frame. This feature can be especially useful when multiple emotions are covered in the same video.
<br>
<br>

![image](https://github.com/Suhail270/fer-may-hackathon/assets/57321434/b0146f09-74a4-4e17-b0cf-3917d8c01072)

<img width="614" alt="image" src="https://github.com/Suhail270/fer-may-hackathon/assets/57321434/f733cf20-1ae5-4cdc-af34-1c3fc831eb11">


#### Implementation Methodology and Rationale

##### Libraries and Imports
The following libraries are used in this project:

- *streamlit* for creating the web application.
- *cv2 (OpenCV)* for image processing and face detection.
- *numpy* for array manipulation.
- *keras* for loading the pre-trained emotion detection model.
- *tempfile* for handling temporary files.
- *streamlit_webrtc* for handling real-time video processing.

##### Model and Classifier Loading
The custom-trained emotion detection model and Haar Cascade face classifier are loaded at the beginning of the script. This ensures that the models are ready for use when processing the video frames.

##### Emotion Counts Initialization
An emotion count dictionary is initialized to keep track of the number of times each emotion is detected in the video frames.

##### Video Transformer Class
A custom VideoTransformer class is defined to process each video frame. This class uses the transform method to:

##### Convert the frame to grayscale.
Detect faces in the frame.
Predict the emotion for each detected face.
Draw a rectangle around each face and annotate it with the predicted emotion label.
The emotion counts are updated for each prediction.

##### Video Processing Function
The process_video function handles the uploaded video file. It:

Saves the uploaded file to a temporary location.
Reads the video frame-by-frame.
Processes each frame to detect faces and emotions.
Stores the processed frames in a list.
Displays the detected emotions and their counts.
Provides a slider to navigate through the frames.

##### Main Function
The main function sets up the Streamlit interface. It sets the title of the app, provides a file uploader for the user to upload a video file and calls the process_video function if a file is uploaded.


# Facial Emotion Recognition

<div id="top"></div>
<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)


</div>

<br />
<div align="center">
    <img src="assets/falcons-logo2.png" alt="Logo" >
</div>
<br /><br />
<div align="center">
    <img src="assets/dds_logo.png" alt="DDS logo" >
</div>
# Decoding Data Science in partnership with Falcons.ai
<br /><br />

Objective: Develop an efficient facial emotion classification system employing OpenCV/Tensorflow to identify facial emotions within video streams. The goal is to achieve a high level of accuracy, low latency, and minimal computational overhead.

Similar to: <br/>

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Msk1drgWEdY/0.jpg)](https://www.youtube.com/watch?v=Msk1drgWEdY)

Data Source: A video dataset or a combination of image datasets featuring the target objects in states of emotion.

Kaggle : https://www.kaggle.com/datasets/msambare/fer2013

Preprocessing (if needed): Standardize or augment the images/video frames to improve model generalization, if necessary, while preserving the aspect ratio and critical features.

Model Selection & Training:
1. Using the FER dataset(partial).
2. Train a custom model using the prepared dataset and analyze the performance.
3. Deploy Streamlit and OpenCV to allow users a web ui in which to upload a video and have the video frames analyzed by your model.

Expecation

The expectations are for the following: 
1) The code used to train the model.
2) The model you trained.
3) The Code used to run the UI and upload the video for inference.

This problem set provides a clear path to address image analysis issues using OpenCV, with a focus on Facial Emotion Classification in video streams. It allows researchers or students to hone in on critical aspects such as data preprocessing, model selection, hyperparameter tuning, performance evaluation, and results interpretation.
    <br /><br />

-------------- Fully functional Jupyternotebook will be added upon hack-a-thon challenge completion  --------------


  </p>
<p align="right">(<a href="#top">back to top</a>)</p>
<br />

<!-- How to use -->
## Usage
<br />
  <p>
   To use the notebook with relative ease please follow the steps below:
    <br />
</p>

1. Ensure all of the required libraries are installed.

2. Load the libraries.

3. Run the cells and the cloud images will be generated and saved in the "clouds" directory.

  </p>
  <br />
<p align="right">(<a href="#top">back to top</a>)</p>
<br />




<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you want, feel free to fork this repository. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/YourFeature`)
3. Commit your Changes (`git commit -m 'Add some YourFeature'`)
4. Push to the Branch (`git push origin feature/YourFeature`)
5. Open a Pull Request
<br />


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

![](https://img.shields.io/badge/License-MIT-blue)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Project Link: [https://github.com/Falcons-ai/fer_dds_challenge]


<p align="right">(<a href="#top">back to top</a>)</p>



Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you want, feel free to fork this repository. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

Fork the Project
Create your Feature Branch (git checkout -b feature/YourFeature)
Commit your Changes (git commit -m 'Add some YourFeature')
Push to the Branch (git push origin feature/YourFeature)
Open a Pull Request
