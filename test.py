# -*- coding: utf-8 -*-

"""
Performance test of CNN architectures on various degradation models - Updated for TensorFlow 2.x / Keras 3.x (2025)

Created on Thu May 24 11:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/cnn-on-degraded-images

Updated for TensorFlow 2.x compatibility
"""

# imports
from __future__ import division
from __future__ import print_function

import cv2
import glob
import json
import numpy as np
import os
import pandas as pd
import sys
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras import applications
from tensorflow.keras.utils import get_custom_objects
from matplotlib import pyplot as plt

# Import custom modules
from libs.CapsuleNetwork import CapsuleLayer, Length, Mask, squash
from libs.DegradationModels import imdegrade
from libs.PipelineUtils import save_samples, shutdown

# configurations
# -----------------------------------------------------------------------------

RANDOM_SEED = None
PROCESS_ID = f"test_{int(time.time())}"

DATASET_ID = 'synthetic_digits'
LABEL_MAPS = f'data/{DATASET_ID}/labelmap.json'
IMAGE_DSRC = f'data/{DATASET_ID}/imgs_valid/'
IMAGE_READ = cv2.IMREAD_COLOR

SAVE_NOISY = False
SAMP_NOISY = 10

NOISE_LIST = ['Gaussian_White', 'Gaussian_Color', 'Salt_and_Pepper',
              'Motion_Blur', 'Gaussian_Blur', 'JPEG_Quality']

MODEL_LIST = ['capsnet', 'mobilenet', 'inceptionv3', 'resnet50', 'vgg16', 'vgg19']

# Updated paths to match your training scripts
MODELS_DICT = {name.lower(): f'output/{DATASET_ID}/{name}/models/{name}.json' 
               for name in MODEL_LIST}

WEIGHT_DICT = {name.lower(): f'output/{DATASET_ID}/{name}/checkpoints/{name}_best.weights.h5' 
               for name in MODEL_LIST}

TOP_N_PRED = 3

OUTPUT_DIR_NOISY = f'output/{DATASET_ID}/__test__images__'
OUTPUT_DIR_TOP_1 = f'output/{DATASET_ID}/__test__top1__/'
OUTPUT_DIR_TOP_N = f'output/{DATASET_ID}/__test__top{TOP_N_PRED}__/'

F_SHUTDOWN = False

# -----------------------------------------------------------------------------

# Setup parameters (reduced ranges for faster CPU testing)
sigmavals = [x for x in range(0, 51, 10)]
densities = [x/100 for x in range(0, 51, 20)]
mb_ksizes = [x for x in range(3, 16, 4)]
gb_ksizes = [x for x in range(1, 26, 8)]
qualities = [x for x in range(30, -1, -10)]

def setup_custom_objects():
    """Setup custom objects for model loading"""
    # Clear existing custom objects
    get_custom_objects().clear()
    
    # Register all custom objects
    custom_objects = {
        'CapsuleLayer': CapsuleLayer,
        'Mask': Mask,
        'Length': Length,
        'squash': squash,
        'relu6': tf.nn.relu6,  # Fixed reference
    }
    
    get_custom_objects().update(custom_objects)
    print('[INFO] Custom objects registered successfully')
    return custom_objects

def validate_paths():
    """Validate all required paths exist"""
    flag = True
    
    if not os.path.isfile(LABEL_MAPS):
        print(f'[INFO] Label mapping not found at {LABEL_MAPS}')
        flag = False
        
    if not os.path.isdir(IMAGE_DSRC):
        print(f'[INFO] Image data source not found at {IMAGE_DSRC}')
        flag = False
        
    # Check available models
    available_models = []
    for name, path in MODELS_DICT.items():
        if os.path.isfile(path) and os.path.isfile(WEIGHT_DICT[name]):
            available_models.append(name)
        else:
            print(f'[INFO] {name} model or weights not found, skipping...')
    
    if not available_models:
        print('[ERROR] No trained models found!')
        flag = False
    else:
        global MODEL_LIST
        MODEL_LIST = available_models
        print(f'[INFO] Available models: {MODEL_LIST}')
    
    # Create output directories
    for directory in [OUTPUT_DIR_NOISY, OUTPUT_DIR_TOP_1, OUTPUT_DIR_TOP_N]:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    
    return flag

def load_data():
    """Load test images and labels"""
    x = []
    y = []
    
    # Load label mapping
    with open(LABEL_MAPS, 'r') as file:
        labelmap = json.load(file)
    
    # Get class labels
    labels = [os.path.split(d[0])[-1] for d in os.walk(IMAGE_DSRC)][1:]
    
    # Read images
    for label in labels:
        image_files = glob.glob(os.path.join(IMAGE_DSRC, label, '*.*'))[:200]  # Limit for faster testing
        for file in image_files:
            image = cv2.imread(file, IMAGE_READ)
            if image is None:
                continue
            x.append(image)
            y.append(labelmap[label])
    
    print(f'[INFO] Loaded {len(x)} images from {len(labels)} classes')
    return (x, y)

def load_models():
    """Load trained models with proper custom objects"""
    models = {}
    custom_objects = setup_custom_objects()
    
    for name in MODEL_LIST:
        try:
            print(f'[INFO] Loading {name}...')
            
            # Load model architecture
            with open(MODELS_DICT[name], 'r') as file:
                model_json = file.read()
            
            # Load with custom objects
            try:
                model = model_from_json(model_json, custom_objects=custom_objects)
            except Exception as e:
                print(f'[WARNING] Failed to load {name} with custom objects: {str(e)}')
                # Try loading without compilation
                model = model_from_json(model_json)
            
            # Load model weights
            try:
                model.load_weights(WEIGHT_DICT[name])
                models[name] = model
                print(f'[INFO] Successfully loaded {name}')
            except Exception as e:
                print(f'[ERROR] Failed to load weights for {name}: {str(e)}')
                continue
            
        except Exception as e:
            print(f'[ERROR] Failed to load {name}: {str(e)}')
            continue
    
    return models

def preprocess_images(images, target_size):
    """Preprocess images for model input"""
    processed = []
    for img in images:
        # Resize image to target size
        resized = cv2.resize(img, target_size)
        # Normalize to [0, 1]
        normalized = resized.astype('float32') / 255.0
        processed.append(normalized)
    
    return np.array(processed)

def test_models(x, y, models):
    """Test models on given images and labels"""
    print('')
    samples = len(x)
    results_top_1 = {}
    results_top_n = {}
    
    for name, model in models.items():
        print(f'[INFO] Testing {name}...')
        
        try:
            # Get input shape from model
            if hasattr(model, 'input_shape') and model.input_shape is not None:
                if isinstance(model.input_shape, list):
                    # Handle multi-input models (like CapsNet)
                    input_shape = model.input_shape[0][1:3]
                else:
                    input_shape = model.input_shape[1:3]
            else:
                # Default fallback
                input_shape = (224, 224)
            
            # Preprocess images
            print(f'[INFO] Preprocessing images for {name} (target size: {input_shape})...')
            x_test = preprocess_images(x, input_shape)
            y_test = np.array(y)
            
            # Make predictions
            print(f'[INFO] Making predictions with {name}...')
            predictions = model.predict(x_test, batch_size=32, verbose=0)
            
            # Handle different output formats
            if isinstance(predictions, list):
                # For models with multiple outputs (like CapsNet)
                predictions = predictions[0]
            
            # Calculate top-k predictions
            p_test = predictions.argsort(axis=1)[:, -TOP_N_PRED:]
            
            # Calculate accuracies
            accuracy_top_1 = np.sum(p_test[:, -1] == y_test) * 100.0 / samples
            accuracy_top_n = np.sum([int(y in p) for y, p in zip(y_test, p_test)]) * 100.0 / samples
            
            results_top_1[f'acc_{name}'] = accuracy_top_1
            results_top_n[f'acc_{name}'] = accuracy_top_n
            
            print(f'[INFO] {name} - Top-1: {accuracy_top_1:.2f}% | Top-{TOP_N_PRED}: {accuracy_top_n:.2f}%')
            
        except Exception as e:
            print(f'[ERROR] Testing {name} failed: {str(e)}')
            results_top_1[f'acc_{name}'] = 0.0
            results_top_n[f'acc_{name}'] = 0.0
            continue
    
    return [results_top_1, results_top_n]

def init_histories(init_dict={}):
    """Initialize result histories"""
    histories_top_1 = init_dict.copy()
    histories_top_n = init_dict.copy()
    
    for name in MODEL_LIST:
        histories_top_1[f'acc_{name.lower()}'] = []
        histories_top_n[f'acc_{name.lower()}'] = []
    
    return [histories_top_1, histories_top_n]

def save_and_plot_histories(file_id, histories, title='', xlabel='', ylabel='',
                           invert_xaxis=False, invert_yaxis=False):
    """Save and plot test histories"""
    for hist_dict, output_dir, ylabel_prefix in zip(
        histories,
        [OUTPUT_DIR_TOP_1, OUTPUT_DIR_TOP_N],
        ['', f'top-{TOP_N_PRED} ']):
        
        # Save histories to CSV
        df = pd.DataFrame(hist_dict)
        df.to_csv(os.path.join(output_dir, f'{file_id}.csv'), index=False)
        
        # Plot histories
        plt.figure(figsize=(12, 8))
        plt.title(f'{title} [Process ID: {PROCESS_ID}]')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel_prefix + ylabel)
        
        if invert_xaxis:
            plt.gca().invert_xaxis()
        if invert_yaxis:
            plt.gca().invert_yaxis()
        
        # Find x-axis data
        x_key = [key for key in hist_dict.keys() if not key.startswith('acc_')][0]
        
        # Plot each model
        for y_key in hist_dict.keys():
            if y_key == x_key:
                continue
            model_name = y_key.split('_')[-1]
            plt.plot(hist_dict[x_key], hist_dict[y_key], 
                    label=model_name, marker='o', linewidth=2)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'{file_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()

# Simplified test function for one degradation type
def test_single_degradation(x, y, models, degradation_type, param_name, param_values):
    """Test a single degradation type"""
    print(f'\n{"="*60}')
    print(f'TESTING {degradation_type.upper().replace("_", " ")}')
    print("="*60)
    
    histories = init_histories({param_name: param_values})
    
    for param_val in param_values:
        print(f'\n[INFO] Testing {degradation_type} ({param_name}={param_val})...')
        
        # Apply degradation
        try:
            noisy = []
            for image in x:
                if degradation_type == 'gaussian_white':
                    degraded = imdegrade(image, 'gaussian_white', mu=0, sigma=param_val, seed=RANDOM_SEED)
                elif degradation_type == 'gaussian_color':
                    degraded = imdegrade(image, 'gaussian_color', mu=0, sigma=param_val, seed=RANDOM_SEED)
                elif degradation_type == 'salt_and_pepper':
                    degraded = imdegrade(image, 'salt_and_pepper', density=param_val, seed=RANDOM_SEED)
                elif degradation_type == 'motion_blur':
                    mb_kernel = np.zeros((param_val, param_val))
                    mb_kernel[param_val//2, :] = 1
                    mb_kernel /= np.sum(mb_kernel)
                    degraded = imdegrade(image, 'motion_blur', mb_kernel=mb_kernel, seed=RANDOM_SEED)
                elif degradation_type == 'gaussian_blur':
                    degraded = imdegrade(image, 'gaussian_blur', gb_ksize=(param_val, param_val), seed=RANDOM_SEED)
                elif degradation_type == 'jpeg_quality':
                    degraded = imdegrade(image, 'jpeg_compression', quality=param_val, seed=RANDOM_SEED)
                else:
                    degraded = image
                noisy.append(degraded)
            
            # Test models
            results = test_models(noisy, y, models)
            
            # Update histories
            for hist_dict, res_dict in zip(histories, results):
                for key in res_dict.keys():
                    hist_dict[key].append(res_dict[key])
                    
        except Exception as e:
            print(f'[ERROR] Failed to apply {degradation_type}: {str(e)}')
            # Add zero results for failed degradation
            for hist_dict in histories:
                for key in hist_dict.keys():
                    if key.startswith('acc_'):
                        hist_dict[key].append(0.0)
    
    # Save and plot results
    invert_x = degradation_type == 'jpeg_quality'
    save_and_plot_histories(
        file_id=degradation_type,
        histories=histories,
        title=f'Accuracy vs {degradation_type.replace("_", " ").title()}',
        xlabel=param_name.replace('_', ' ').title(),
        ylabel='Accuracy (%)',
        invert_xaxis=invert_x
    )

def test():
    """Main testing function"""
    print('='*80)
    print('CNN ARCHITECTURE ROBUSTNESS TESTING')
    print('='*80)
    print(f'Process ID: {PROCESS_ID}')
    print(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*80)
    
    # Validate paths
    if not validate_paths():
        print('[ERROR] Path validation failed. Exiting.')
        return
    
    # Load test data
    print('\n[INFO] Loading test data...')
    try:
        (x, y) = load_data()
        print(f'[INFO] Loaded {len(x)} test samples')
    except Exception as e:
        print(f'[ERROR] Failed to load data: {str(e)}')
        return
    
    # Load models
    print('\n[INFO] Loading trained models...')
    try:
        models = load_models()
        if not models:
            print('[ERROR] No models loaded successfully. Exiting.')
            return
        print(f'[INFO] Loaded {len(models)} models: {list(models.keys())}')
    except Exception as e:
        print(f'[ERROR] Failed to load models: {str(e)}')
        return
    
    print('\n' + '='*80)
    print('BEGINNING ROBUSTNESS TESTS')
    print('='*80)
    
    # Test on clean images first
    print('\n[INFO] Testing on clean (original) images...')
    try:
        clean_results = test_models(x, y, models)
        
        # Save clean results
        for results, output_dir, prefix in zip(
            clean_results,
            [OUTPUT_DIR_TOP_1, OUTPUT_DIR_TOP_N], 
            ['', f'top{TOP_N_PRED}_']):
            
            df = pd.DataFrame([results])
            df.to_csv(os.path.join(output_dir, f'{prefix}clean_results.csv'), index=False)
    except Exception as e:
        print(f'[ERROR] Clean image testing failed: {str(e)}')
    
    # Run degradation tests
    degradation_tests = [
        ('gaussian_white', 'sigma', sigmavals),
        ('gaussian_color', 'sigma', sigmavals),
        ('salt_and_pepper', 'density', densities),
        ('motion_blur', 'kernel_size', mb_ksizes),
        ('gaussian_blur', 'kernel_size', gb_ksizes),
        ('jpeg_quality', 'image_quality', qualities)
    ]
    
    for degradation_type, param_name, param_values in degradation_tests:
        try:
            test_single_degradation(x, y, models, degradation_type, param_name, param_values)
        except Exception as e:
            print(f'[ERROR] Test {degradation_type} failed: {str(e)}')
            continue
    
    print('\n' + '='*80)
    print('ALL TESTS COMPLETED')
    print('='*80)
    print(f'End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Results saved in: {OUTPUT_DIR_TOP_1} and {OUTPUT_DIR_TOP_N}')
    print('='*80)

def main():
    """Main entry point with error handling"""
    try:
        test()
        if F_SHUTDOWN:
            shutdown()
    except KeyboardInterrupt:
        print('\n[INFO] Process interrupted by user')
    except Exception as e:
        error = sys.exc_info()[0].__name__ if sys.exc_info()[0] is not None else 'Unknown'
        print(f'\n[ERROR] Process failed: {error} - {str(e)}')
    finally:
        print('[INFO] Testing session ended')

if __name__ == '__main__':
    main()
