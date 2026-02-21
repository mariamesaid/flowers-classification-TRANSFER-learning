# flowers-classification-TRANSFER-learning
The following steps were taken to develop this image classification project using the tf_flowers dataset:
Environment Setup and Hardware Check: The project initializes by importing necessary libraries (TensorFlow, TensorFlow Hub, Matplotlib, and NumPy) and verifying the availability of a GPU to ensure efficient model training.

Data Loading and Splitting: The tf_flowers dataset is loaded using TensorFlow Datasets. It is split into two subsets: 80% for training (train_ds) and 20% for validation (val_ds).Data Preprocessing:Images are resized to $224 \times 224$ pixels to match the input requirements of the chosen neural network.Pixel values are normalized by dividing by 255.0 to scale them between 0 and 1.The training data is shuffled (buffer size 1000), batched into groups of 32, and prefetched to optimize pipeline performance.

Model Architecture Construction:Base Model: The project utilizes transfer learning by employing a pre-trained MobileNetV2 model (trained on ImageNet) as a feature extractor.
The top (classification) layer is excluded, and the weights are frozen (trainable = False).Custom Layers: A Sequential model is built by adding a GlobalAveragePooling2D layer, a Dropout layer (0.2) to prevent overfitting, and a final Dense layer with a softmax activation function to classify the images into the five flower categories.Model Compilation:

The model is compiled using the Adam optimizer and sparse categorical crossentropy as the loss function. Accuracy is selected as the primary performance metric.Training: The model is trained for 5 epochs using the training batches, with the validation batches used to monitor performance at each step. By the fifth epoch, the model achieved a training accuracy of approximately 91% and a validation accuracy of approximately 89.6%.
Evaluation and Visualization: After training, the model makes predictions on a batch from the validation set. The results are visualized using Matplotlib, showing the images alongside their predicted and actual labels, with green text indicating correct predictions and red for incorrect ones.
