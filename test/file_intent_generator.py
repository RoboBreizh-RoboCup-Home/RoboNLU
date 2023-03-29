from typing import List

from predict_robot import CommandProcessor
import ast
import os


def read_input_file(input_text_path: str) -> List[List[str]]:
    word_lines = []
    with open(input_text_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            # word_lines.append(line.strip().replace(", "," , ").split())
            word_lines.append(line.strip().split())
    return word_lines


def write_output_files(word_lines: List[str], model: CommandProcessor, gpsr_name: str):
    for index, words in enumerate(word_lines):
        intent_list: List[str] = model.predict(words).split("\n")
        for intent in intent_list:
            input_dict = ast.literal_eval(intent)
            # retrieve attribute associated with intent key
            intent_attr = input_dict.get('intent')
            with open("output/"+intent_attr+"_"+gpsr_name+".out", "a", encoding="utf-8") as f:
                f.write(' '.join(word_lines[index]) + '\n    ' + intent + '\n')


if __name__ == "__main__":
    # removes old test output
    test_files = os.listdir('./output')
    for test_file in test_files:
        if test_file.endswith(".out"):
            os.remove("./output/"+test_file)
    # generate gpsr output
    word_lines: List[List[str]] = read_input_file("./input/gpsr.in")
    model_path = 'quantized_models/bert.quant.onnx'
    slot_classifier_path = 'numpy_para/slot_classifier'
    intent_token_classifier_path = 'numpy_para/intent_token_classifier'
    pro_classifier_path = 'numpy_para/pro_classifier'
    model = CommandProcessor(model_path=model_path, slot_classifier_path=slot_classifier_path,
                             intent_token_classifier_path=intent_token_classifier_path,
                             pro_classifier_path=pro_classifier_path)
    write_output_files(word_lines, model, "gpsr")

    # generate egpsr output
    word_lines: List[List[str]] = read_input_file("./input/egpsr.in")
    model = CommandProcessor(model_path=model_path, slot_classifier_path=slot_classifier_path,
                             intent_token_classifier_path=intent_token_classifier_path,
                             pro_classifier_path=pro_classifier_path)
    write_output_files(word_lines, model, "egpsr")
