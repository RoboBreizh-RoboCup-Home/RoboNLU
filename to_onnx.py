from numpy import save
import os

from utils import MODEL_CLASSES

import torch
import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic
import onnxruntime

import argparse
import logging

logging.getLogger().setLevel(logging.INFO)

def main(args):
    save_path = args.out_path
    model_type = args.model_type

    # Load the model
    logging.info(f"Loading model from {args.model_path}...")
    model = torch.load(args.model_path)
    model.eval()
    model.to('cpu')

    numpy_para = os.path.join(save_path, 'numpy_para')
    if not os.path.exists(numpy_para):
        os.makedirs(numpy_para)
        os.makedirs(os.path.join(numpy_para, 'slot_classifier'))
        os.makedirs(os.path.join(numpy_para, 'intent_token_classifier'))
        os.makedirs(os.path.join(numpy_para, 'pro_classifier'))

    # export classifiers weights
    logging.info(f"Saving classifiers weights to {numpy_para}...")
    slot_classifier_w = model.slot_classifier._modules['linear'].weight.detach().cpu().numpy()
    save(f'{numpy_para}/slot_classifier/weights.npy', slot_classifier_w)
    slot_classifier_b = model.slot_classifier._modules['linear'].bias.detach().cpu().numpy()
    save(f'{numpy_para}/slot_classifier/bias.npy', slot_classifier_b)

    intent_token_classifier_w = model.intent_token_classifier._modules['linear'].weight.detach().cpu().numpy()
    save(f'{numpy_para}/intent_token_classifier/weights.npy', intent_token_classifier_w)
    intent_token_classifier_b = model.intent_token_classifier._modules['linear'].bias.detach().cpu().numpy()
    save(f'{numpy_para}/intent_token_classifier/bias.npy', intent_token_classifier_b)

    pro_classifier_w1 = model.pro_classifier._modules['linear1'].weight.detach().cpu().numpy()
    save(f'{numpy_para}/pro_classifier/weights1.npy', pro_classifier_w1)
    pro_classifier_b1 = model.pro_classifier._modules['linear1'].bias.detach().cpu().numpy()
    save(f'{numpy_para}/pro_classifier/bias1.npy', pro_classifier_b1)

    pro_classifier_w2 = model.pro_classifier._modules['linear2'].weight.detach().cpu().numpy()
    save(f'{numpy_para}/pro_classifier/weights2.npy', pro_classifier_w2)
    pro_classifier_b2 = model.pro_classifier._modules['linear2'].bias.detach().cpu().numpy()
    save(f'{numpy_para}/pro_classifier/bias2.npy', pro_classifier_b2)

    # dummy inputs
    dummy_input = {
        'input_ids': torch.randint(0, 30522, (1, 32), dtype=torch.long),
        'attention_mask': torch.ones(1, 32, dtype=torch.long),
        'token_type_ids': torch.zeros(1, 32, dtype=torch.long)
    }

    logging.info(f"Exporting model to onnx, fp32 model saved at {save_path}...")
    fp32_model_path = os.path.join(save_path, args.model_type+'_fp32.onnx')

    inputs=['input_ids','attention_mask','token_type_ids']

    if model_type == 'distilbert':
        to_export = model.distilbert
        inputs=['input_ids','attention_mask']
        dummy_input = {
            'input_ids': torch.randint(0, 30522, (1, 32)),
            'attention_mask': torch.ones(1, 32),
        }
    elif model_type == 'mobilebert':
        to_export = model.mobilebert
    elif model_type == 'bert':
        to_export = model.bert
    elif model_type == 'albert':
        to_export = model.albert

    torch.onnx.export(to_export.to('cpu'),               # model being run
                    dummy_input,               # model input (or a tuple for multiple inputs)
                    fp32_model_path,            # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=17,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = inputs,   # the model's input names
                    output_names = ['sequence_output'], ##['all_logits','other'],
                    )

    fp16_model_path = os.path.join(save_path, args.model_type+'_fp16.onnx')

    model = onnx.load(fp32_model_path)

    if args.fp16:
        logging.info(f"Converting model to fp16, fp16 model saved at {save_path}...")
        from onnxconverter_common import float16

        model_fp16 = float16.convert_float_to_float16(model)

        onnx.save(model_fp16, fp16_model_path)

    if args.int8:
        logging.info(f"Converting model to int8, int8 model saved at {save_path}...")
        int8_model_path = os.path.join(save_path, args.model_type+'_int8.onnx')

        quantize_dynamic(fp32_model_path, int8_model_path)


    # load and test the .onnx file
    onnx_model = onnx.load(fp32_model_path)
    onnx.checker.check_model(onnx_model)

    logging.info(f"Testing the exported model...")
    ort_session = onnxruntime.InferenceSession(fp32_model_path)
    
    random_input = torch.randint(0, 30522, (1, 32))
    attention_mask = torch.ones(random_input.shape)

    if model_type == 'distilbert':
        ort_inputs = {
            'input_ids': random_input.numpy(),
            'attention_mask': attention_mask.numpy(),
        }
    else:
        ort_inputs = {
            'input_ids': random_input.numpy().astype(np.int64),
            'attention_mask': attention_mask.numpy().astype(np.int64),
            'token_type_ids': torch.zeros(random_input.shape).numpy().astype(np.int64)
        }

    try:
        ort_outs = ort_session.run(None, ort_inputs)
        assert len(ort_outs) != 0
    except Exception as e:
        logging.error(f"Error, model export failed: {e}")

    logging.info(f"Model exported successfully to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--model_type", default=None, required=True, type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--out_path", default=None, required=True, type=str, help="Path to save model in onnx and additional classifiers in npy")
    parser.add_argument("--fp16", action='store_true', help="Whether to convert to fp16 or not")
    parser.add_argument("--int8", action='store_true', help="Whether to convert to int8 or not")

    args = parser.parse_args()
    main(args)
