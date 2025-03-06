# Downscaling Climate Data using Machine Learning
# Published Papaer Name: Statistical Downscaled Climate Projection Dataset for China Using Artificial Neural Networks

Scientific question: How to combine the two types of data to obtain more accurate climate change forecasts for the future?

* Overview

  This project applies statistical downscaling techniques to enhance climate data resolution using machine learning (MLPRegressor - Artificial Neural Networks). The primary focus is on temperature and precipitation downscaling, improving the accuracy of coarse-grained climate models for regional climate studies.

* Motivation

  Global Climate Models (GCMs) provide climate projections at a coarse resolution, which is insufficient for localized climate impact studies. Downscaling bridges this gap by using machine learning to predict high-resolution climate data from low-resolution inputs.

* Features

  ✅ ML-based Statistical Downscaling using MLPRegressor

  ✅ Training on Climate Datasets (e.g., historical temperature & precipitation data)

  ✅ Improving Spatial & Temporal Resolution

  ✅ Model Evaluation & Visualization

  ✅ Python Implementation with Scikit-Learn & Pandas

* Technologies Used

  Python (NumPy, Pandas, Scikit-Learn, Matplotlib)

  MLPRegressor (Multi-layer Perceptron for regression tasks)

  Data Processing & Feature Engineering

* Installation: 

  1. Clone the repository: 

      git clone https://github.com/Kepler22b22/Downscaling.git
  
      cd Downscaling

  2. Install dependencies:
 
     pip install -r requirements.txt

* Usage:

  1. Prepare your dataset (ensure it includes historical climate data).

  2. Run the training script:

      python train.py --data <path_to_data>

  3. Evaluate the model:

      python evaluate.py --model <path_to_model>

  4. Visualize results:

      python visualize.py

* Results

  · The trained model significantly improves the spatial resolution of climate data.

  · Visualizations demonstrate the enhanced temperature and precipitation predictions.

* Contributions

  Contributions are welcome! Feel free to open issues and submit pull requests.
