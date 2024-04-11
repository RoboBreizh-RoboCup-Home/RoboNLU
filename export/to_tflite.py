# this script export a pytorch model to tflite

import os
import torch
import numpy as np
import argparse
import onnx2tf
# pip install onnx2tf
import tensorflow as tf

def save(file, data):
    with open(file, 'wb') as f:
        np.save(f, data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--onnx_path", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--model_type", default="distilbert", required=True, type=str, help="Model type")

    args = parser.parse_args()

    # load pytorch model
    model_path = args.onnx_path
    model_type = args.model_type

    # get directory of onnx_path
    save_path = os.path.dirname(model_path)
    save_path = os.path.join(save_path, model_type+'/tflite/')
    os.makedirs(save_path, exist_ok=True)

    onnx2tf.convert(
        input_onnx_file_path=model_path,
        output_folder_path=save_path,
        copy_onnx_input_output_names_to_tflite=True,
        verbose=True)
