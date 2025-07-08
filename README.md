# Image-Classification-Using-Convolutional-Neural-Networks

This project focuses on building and evaluating a Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset. The goal is to gain a solid understanding of CNN architecture, improve model performance through data augmentation, and deploy the final trained model efficiently.

Problem Statement : Build and evaluate a CNN to classify images from a publicly available dataset (MNIST).
The aim is to achieve high accuracy while understanding CNN architecture and its real-world applications.

Tools / Libraries : Python, TensorFlow, Keras, Matplotlib, Seaborn, TensorBoard, MNIST Dataset (via tensorflow.keras.datasets), AWS

Skills Gained :
- Deep Learning fundamentals
- CNN architecture design
- Image preprocessing and augmentation
- Model evaluation and performance tuning
- Model deployment and reproducibility practices

Project Approach :
1. Dataset Loading and Exploration : Imported MNIST dataset using tf.keras.datasets. Visualized sample images and class distribution using Matplotlib.

2. Data Preprocessing : Normalized pixel values (0–1). Converted labels to one-hot encoding. Split dataset into training, validation, and test sets.

3. Data Augmentation : Applied augmentation techniques like rotation, shift, zoom, shear using ImageDataGenerator.

4. Build the CNN Model : Constructed CNN with Conv2D, ReLU, MaxPooling, Flatten, Dense. Used ReLU as activation, Adam as optimizer.

5. Compile and Train the Model : Used categorical_crossentropy as Loss Function. Used EarlyStopping and ReduceLROnPlateau as Callbacks.
Performance Monitoring using TensorBoard.

6. Evaluate the Model
Metrics used: Accuracy, Precision, Recall, F1 Score
Visualization: Confusion Matrix, Accuracy/Loss Curves

7. Save and Deploy the Model : Saved final model as .h5 file. Model can be deployed on AWS using Flask/Streamlit

Evaluation Metrics : Accuracy, Precision, Recall, F1 Score, Binary Cross-Entropy Loss, Confusion Matrix
