# Web Application Setup Guide

This guide explains how to run the CIFAR-10 Image Classifier web application.

## Files Overview

- **index.html** - Interactive web interface for image prediction
- **app.py** - Flask backend server that handles model predictions
- **requirements.txt** - Python dependencies

## Setup Instructions

### Step 1: Save Your Trained Model

First, run your Jupyter notebook to completion and save the model:

1. Open `ClassificationCIFAR-10.ipynb`
2. Run all cells (especially the last cell "Sauvegarder le modèle")
3. This will create a `cifar10_model.h5` file

### Step 2: Install Dependencies

Open PowerShell and run:

```powershell
cd C:\Users\salim\Desktop\cifar
pip install -r requirements.txt
```

### Step 3: Run the Server

In PowerShell:

```powershell
python app.py
```

You should see:
```
==================================================
Loading CIFAR-10 Classification Model
==================================================
✓ Model loaded from cifar10_model.h5
✓ Starting Flask server...
  Open http://localhost:5000 in your browser
==================================================
```

### Step 4: Use the Web App

1. Open your web browser
2. Go to `http://localhost:5000`
3. Upload an image (32x32 or larger)
4. The model will predict the class and show confidence

## Features

✓ **Drag & Drop Upload** - Simply drag images onto the upload area  
✓ **Real-time Prediction** - Get instant results  
✓ **Confidence Display** - See how confident the model is  
✓ **All Predictions** - View probabilities for all 10 classes  
✓ **Visual Feedback** - Beautiful, responsive UI  

## Supported Classes

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Troubleshooting

**Issue**: "Model file not found"
- **Solution**: Make sure you ran the notebook and the `cifar10_model.h5` file was created

**Issue**: "Flask not found"
- **Solution**: Run `pip install Flask` in PowerShell

**Issue**: "Port 5000 is already in use"
- **Solution**: Edit `app.py` and change `app.run(debug=True, port=5000)` to a different port number

**Issue**: "TensorFlow not found"
- **Solution**: Run `pip install tensorflow` in PowerShell

## Project Structure

```
cifar/
├── ClassificationCIFAR-10.ipynb  (Jupyter notebook - train the model)
├── cifar10_model.h5              (Trained model - created after running notebook)
├── index.html                    (Web interface)
├── app.py                        (Flask backend)
├── requirements.txt              (Dependencies)
├── README.md                     (Project description)
└── WEBAPP_SETUP.md              (This file)
```

## Next Steps

- Improve model accuracy by training for more epochs
- Add data augmentation for better generalization
- Deploy on a cloud platform (Heroku, AWS, Google Cloud)
- Add more image preprocessing options
- Implement batch prediction
