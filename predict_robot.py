import time
import onnxruntime
import numpy as np
from transformers import BertTokenizer
# from onnxruntime_tools import optimizer

from transformers import AutoTokenizer
from numpy import load


class slot_classifier_np():
    def __init__(self, dir):
        self.linear_weights = load(dir+'/weights.npy')
        self.linear_bias = load(dir+'/bias.npy')

    def ReLU(self, x):
        return x * (x > 0)

    def forward(self, x):
        x = np.squeeze(x)
        x = self.ReLU(x)
        x = x @ np.transpose(self.linear_weights) + self.linear_bias
        return x


class intent_token_classifier_np():
    def __init__(self, dir):
        self.linear_weights = load(dir + '/weights.npy')
        self.linear_bias = load(dir + '/bias.npy')

    def ReLU(self, x):
        return x * (x > 0)

    def forward(self, x):
        x = np.squeeze(x)
        x = self.ReLU(x)
        x = x @ np.transpose(self.linear_weights) + self.linear_bias
        return x


class pro_classifier_np():
    def __init__(self, dir):
        self.linear_weights1 = load(dir + '/weights1.npy')
        self.linear_bias1 = load(dir + '/bias1.npy')
        self.linear_weights2 = load(dir + '/weights2.npy')
        self.linear_bias2 = load(dir + '/bias2.npy')

    def ReLU(self, x):
        return x * (x > 0)

    def forward(self, x):
        x = np.squeeze(x)
        x = x @ np.transpose(self.linear_weights1) + self.linear_bias1
        x = self.ReLU(x)
        x = x @ np.transpose(self.linear_weights2) + self.linear_bias2
        return x


class CommandProcessor(object):
    def __init__(self, session=None):
        # self.model_name = 'bert'
        # self.tokenizer_name = 'bert-base-uncased'
        self.model_name =  'mobile_bert'


        if self.model_name == 'bert':
            self.tokenizer_name = 'bert-base-uncased'
        elif self.model_name == 'mobile_bert':
            self.tokenizer_name = 'google/mobilebert-uncased'
        elif self.model_name == 'distil_bert':
            self.tokenizer_name = 'distilbert-base-uncased'
        # self.INTENT_CLASSES = ['PAD','O','B-greet','I-greet','B-guide','I-guide','B-follow','I-follow','B-find','I-find','B-take','I-take','B-go','I-go','B-know','I-know']
        # self.SLOT_CLASSES = ['PAD','O','B-obj','B-dest','I-sour','I-obj','I-dest','B-per','B-sour','I-per']

        self.INTENT_CLASSES = ['PAD', 'O', 'B-greet', 'I-greet', 'B-know', 'I-know', 'B-follow', 'I-follow', 'B-take',
                               'I-take', 'B-tell', 'I-tell', 'B-guide', 'I-guide', 'B-go', 'I-go', 'B-answer',
                               'I-answer', 'B-find', 'I-find']
        self.SLOT_CLASSES = ['PAD', 'O', 'I-obj', 'B-sour', 'B-dest', 'I-sour', 'B-what', 'B-obj', 'I-dest', 'I-per',
                             'I-what', 'B-per']

        self.PRO_CLASSES = ['PAD', 'O', 'B-referee']

        self.referee_token_map = {i: label for i,
                                  label in enumerate(self.PRO_CLASSES)}
        self.intent_token_map = {i: label for i,
                                 label in enumerate(self.INTENT_CLASSES)}
        self.slot_label_map = {i: label for i,
                               label in enumerate(self.SLOT_CLASSES)}

        self.max_seq_len = 32
        self.pro_lst = ['him', 'her', 'it', 'its']
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.pad_token_label_id = 0

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.input_text_path = './sample_pred_in.txt'
        self.output_file = './outputs'

        self.bert_ort_session = self.initONNX(
            f'./quantized_models/{self.model_name}.quant.onnx')
        # self.slot_classifier_ort_session = self.initONNX('./quantized_models/slot_classifier.quant.onnx')
        # self.intent_token_classifier_ort_session = self.initONNX('./quantized_models/intent_token_classifier.quant.onnx')
        # self.pro_classifier_ort_session = self.initONNX('./quantized_models/pro_classifier.quant.onnx')
        self.slot_classifier = slot_classifier_np(
            f'numpy_para/{self.model_name}/slot_classifier')
        self.intent_token_classifier = intent_token_classifier_np(
            f'numpy_para/{self.model_name}/intent_token_classifier')
        self.pro_classifier = pro_classifier_np(f'numpy_para/{self.model_name}/pro_classifier')

    def initONNX(self, path):
        start = time.time()
        sess_options = onnxruntime.SessionOptions()

        sess_options.intra_op_num_threads = 1  # 4
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        ort_session = onnxruntime.InferenceSession(
            path, sess_options, providers=['CPUExecutionProvider'])
        print("Loading time ONNX: ", time.time() - start)
        return ort_session

    def read_input_file(self):
        with open(self.input_text_path, "r", encoding="utf-8") as f:
            words = f.readline().strip().split()
        return words

    def convert_input_file_to_dataloader(self, words,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):

        tokenizer = self.tokenizer

        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        tokens = []
        slot_label_mask = []
        pro_labels_ids = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            if word in self.pro_lst:
                pro_label = 1
            else:
                pro_label = 0
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend(
                [self.pad_token_label_id + 1] + [self.pad_token_label_id] * (len(word_tokens) - 1))
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            pro_labels_ids.extend(
                [pro_label] + [self.pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > self.max_seq_len - special_tokens_count:
            tokens = tokens[: (self.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(
                self.max_seq_len - special_tokens_count)]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!
            pro_labels_ids = pro_labels_ids[:(
                self.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [self.pad_token_label_id]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pro_labels_ids += [self.pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [self.pad_token_label_id] + slot_label_mask
        pro_labels_ids = [self.pad_token_label_id] + \
            pro_labels_ids  # !!!!!!!!!!!!!!!!!!!!!!!
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + \
            ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + \
            ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + \
            ([self.pad_token_label_id] * padding_length)
        pro_labels_ids = pro_labels_ids + \
            ([self.pad_token_label_id] * padding_length)

        input_ids = np.array(input_ids).astype('int64')
        attention_mask = np.array(attention_mask).astype('int64')
        token_type_ids = np.array(token_type_ids).astype('int64')
        pro_labels_ids = np.array(pro_labels_ids).astype('int64')
        sample = {'input_ids': input_ids[None, :], 'attention_mask': attention_mask[None,
                                                                                    :], 'token_type_ids': token_type_ids[None, :]}

        if self.model_name == 'distil_bert':
            sample = {'input_ids': input_ids[None, :], 'attention_mask': attention_mask[None,:]}

        return sample, slot_label_mask, pro_labels_ids

    def predict(self):
        lines = self.read_input_file()
        # while True:
        #     command = input('please enter a command \n')
        sample, slot_label_mask, pro_labels_ids = self.convert_input_file_to_dataloader(
            lines)


        start = time.time()

        if self.model_name == 'distil_bert':
            sequence_output = self.bert_ort_session.run(None, sample)
        else:
            sequence_output, _ = self.bert_ort_session.run(None, sample)
        # ============================= Slot prediction ==============================
        slot_logits = self.slot_classifier.forward(sequence_output)
        # slot_logits = self.slot_classifier_ort_session.run(None, {'sequence_output':sequence_output})
        slot_preds = np.squeeze(np.array(slot_logits))
        slot_preds = np.argmax(slot_preds, axis=1)

        # ============================== Intent Token Seq =============================
        intent_token_logits = self.intent_token_classifier.forward(
            sequence_output)
        # intent_token_logits = self.intent_token_classifier_ort_session.run(None, {'sequence_output':sequence_output})
        intent_token_preds = np.squeeze(np.array(intent_token_logits))
        intent_token_preds = np.argmax(intent_token_preds, axis=1)

        # ============================= Pronoun referee prediction ==============================
        if any(pro_labels_ids):
            sq_sequence_output = np.squeeze(sequence_output)
            pro_token = sq_sequence_output[pro_labels_ids == 1]

            # gpsr has 2 pronouns referred to the same referral, we only need to encode one pronoun
            if pro_token.shape[0] != 1:
                pro_token = pro_token[0, :]

            repeat_pro = np.tile(pro_token, (self.max_seq_len, 1))
            concated_input = np.concatenate(
                (sq_sequence_output, repeat_pro), axis=1)[None, :]

            referee_token_logits = self.pro_classifier.forward(concated_input)
            # referee_token_logits = self.pro_classifier_ort_session.run(None, {'concated_input':concated_input})
            referee_preds = np.squeeze(np.array(referee_token_logits))
            referee_preds = np.argmax(referee_preds, axis=1)

        else:
            referee_preds = np.ones(self.max_seq_len)

        print('------------------------------------------------------')
        print("Total inference time: ", time.time() - start)

        slot_preds_list = []
        intent_token_preds_list = []
        referee_preds_list = []

        for token_idx in range(len(slot_label_mask)):
            if slot_label_mask[token_idx] != self.pad_token_label_id:
                referee_preds_list.append(
                    self.referee_token_map[referee_preds[token_idx]])
                intent_token_preds_list.append(
                    self.intent_token_map[intent_token_preds[token_idx]])
                slot_preds_list.append(
                    self.slot_label_map[slot_preds[token_idx]])

        self.write_readable_outputs(
            slot_preds_list, intent_token_preds_list, referee_preds_list)

        return

    def write_readable_outputs(self, slot_preds_list, intent_token_preds_list, referee_preds_list):
        words = self.read_input_file()
        line = ''
        for token_idx, (word, i_pred, s_pred, r_pred) in enumerate(zip(words, intent_token_preds_list, slot_preds_list, referee_preds_list)):
            if s_pred == 'O' and i_pred == 'O' and r_pred == 'O':
                line = line + word + " "
            elif i_pred != 'O':
                if word not in self.pro_lst:
                    line = line + "[{}:{}:{}] ".format(word, i_pred, s_pred)

                    if r_pred == 'B-referee':
                        r_idx = token_idx

                else:
                    line = line + "[{}({}):{}:{}] ".format(word,
                                                           words[r_idx], i_pred, s_pred)

        with open(self.output_file, "a", encoding="utf-8") as f:
            if 'B-referee' in referee_preds_list:
                f.write('\n')
                f.write(
                    '---------------------------------------------------------------------\n')
                f.write('* Pro Case: \n')
                f.write(line.strip()+'\n')
                f.write(
                    '---------------------------------------------------------------------\n \n')
            else:
                f.write(line.strip()+'\n')
            print(line)
            print('=====================================')
        return


if __name__ == "__main__":
    inference = CommandProcessor()
    inference.predict()
