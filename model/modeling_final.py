import torch
import torch.nn as nn
# from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from transformers import MobileBertConfig, MobileBertModel
# from torchcrf import CRF
from TorchCRF import CRF

# from .module import IntentClassifier, SlotClassifier, IntentTokenClassifier, MultiIntentClassifier, TagIntentClassifier, ProClassifier
import logging

from transformers import BertConfig

logger = logging.getLogger()

class ProClassifier(nn.Module):
    def __init__(self, input_dim):
        #input_dim is config.hidden_size*2, which is 768*2 for BERT base (*2->cat)
        super(ProClassifier, self).__init__()
        # self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(input_dim, input_dim//2)
        self.linear2 = nn.Linear(input_dim//2, 3)

    def forward(self, x):
        # x = self.dropout(x)

        x = nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

##############################################################################

class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.2):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

###############################################################################
class IntentTokenClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentTokenClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

###############################################################################
class JointBERTMultiIntent(nn.Module):
    def __init__(self, mobile=True):
        super().__init__()
        self.num_intent_labels = 16 #len(intent_label_lst)
        self.num_slot_labels = 10 #len(slot_label_lst)
        self.max_seq_len = 32

        # load pretrain bert
        if mobile:
            self.config = MobileBertConfig.from_pretrained('google/mobilebert-uncased')
            self.bert = MobileBertModel(config=self.config)
            print("Using MobileBert")
        else:
            self.config = BertConfig.from_pretrained('bert-base-uncased')
            self.bert = BertModel(config=self.config)

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                    param.requires_grad = True


        self.slot_classifier = SlotClassifier(self.config.hidden_size, self.num_slot_labels, dropout_rate = 0.1)
        self.intent_token_classifier = IntentTokenClassifier(self.config.hidden_size, self.num_intent_labels, dropout_rate = 0.1)
        self.pro_classifier = ProClassifier(2 * self.config.hidden_size)


    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                pro_labels_ids):  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! adding in pro data


        
        # (len_seq, batch_size, hidden_dim), (batch_size, hidden_dim)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        # B * L * D
        sequence_output = outputs[0]

            
        # ==================================== 2. Slot Softmax ========================================
        # (batch_size, seq_len, num_slots)
        slot_logits = self.slot_classifier(sequence_output)


        # ==================================== NEW. Pronoun ========================================
        referee_token_loss = 0.0
        if  1 in pro_labels_ids:  # if use pro, and the batch contain pronouns
            # 1. concate pronoun to each word in the sequence
            pro_token_mask = pro_labels_ids > 0
            pro_sample_mask = torch.max(pro_token_mask.long(),dim = 1)[0] > 0
            pro_vec = sequence_output[pro_token_mask]
            pro_sequence_output = sequence_output[pro_sample_mask]
            pro_vec = pro_vec[:, None, :]  # add new dimention
            repeat_pro = pro_vec.repeat(1, self.max_seq_len, 1)
            concated_input = torch.cat((pro_sequence_output, repeat_pro), dim=2)

            # 2. feed data into network and accumulate the loss
            referee_token_logits = self.pro_classifier(concated_input)


            # for the prediction, we need the pro labels for all samples
            full_referee_token_logits = torch.zeros((pro_labels_ids.shape[0], referee_token_logits.shape[1],referee_token_logits.shape[2]))
            counter = 0
            for idx,sample_bool in enumerate(pro_sample_mask):
                if sample_bool:
                    full_referee_token_logits[idx] = referee_token_logits[counter]
                    counter += 1

        
        # ==================================== 3. Intent Token Softmax ========================================

            # (batch_size, seq_len, num_intents)
        intent_token_logits = self.intent_token_classifier(sequence_output)


        outputs = (( slot_logits, intent_token_logits, referee_token_logits,full_referee_token_logits),) + outputs[2:]




        return outputs