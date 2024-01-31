
# Optimizing Neural Networks Hyperparameter Tuning with Keras

## Overview
This project tackles the challenge of classifying vehicles into emergency and non-emergency categories. It demonstrates how to build and optimize a neural network using the Keras library. The dataset consists of images of vehicles, which are classified to aid in emergency response management.

## Dataset
The dataset includes a CSV file with classifications and image files corresponding to each entry.

- `emergency_classification.csv`: Contains the labels for each image.
- `images/`: Directory containing vehicle images used for the classification task.

## Environment Setup
To run this project, you need to install the required Python libraries listed in `requirements.txt`.

You can install them using the following command:
```bash
pip install -r requirements.txt
```

## File Structure
- `data/`: Contains the CSV file with the emergency vehicle classification data.
- `myenv/`: Python virtual environment directory.
- `source/`: Contains the source code for the neural network in Keras.
- `images/`: Contains the images for the classification task.
- `nnkeras.py`: The main Python script with the neural network code.

## How to Run
1. Activate your Python virtual environment.
2. Execute the `nnkeras.py` script to start the training process.

```bash
python source/nnkeras.py
```

## Steps Involved
1. Import libraries and load the dataset.
2. Pre-process the data by resizing and normalizing images.
3. Split the dataset into training and validation sets.
4. Define the neural network architecture.
5. Compile the model.
6. Train the model on the training data.
7. Evaluate the model performance on the validation set.
8. Perform hyperparameter tuning to optimize the neural network.

## Hyperparameter Tuning
The project explores several hyperparameter tuning strategies:
- Changing the activation function of hidden layers.
- Increasing the number of neurons in hidden layers.
- Adding more hidden layers.
- Increasing the number of training epochs.
- Switching the optimizer.

Each change is documented and the impact on model performance is evaluated.

## Results
The training process outputs the model's loss and accuracy over the epochs. The results are visualized in graphs for easy comparison and analysis.

## Requirements
- numpy==1.19.5
- pandas==1.1.5
- matplotlib==3.2.2
- keras==2.4.3
- scikit-learn==0.24.1

Make sure to match these versions to avoid any compatibility issues.

## Conclusion
This project demonstrates the effectiveness of different hyperparameter tuning techniques in improving the performance of neural networks.

For more details, refer to the `steps.md` in the source directory.
