import torch
import torch.nn as nn
# from transformers.modeling_distilbert import DistilBertPreTrainedModel, DistilBertModel, DistilBertConfig
from transformers.models.mobilebert.modeling_mobilebert import  MobileBertModel, MobileBertConfig

from .module import IntentClassifier, SlotClassifier, IntentTokenClassifier, MultiIntentClassifier, TagIntentClassifier, ProClassifier

class JointMobileBERTMultiIntent(MobileBertModel):
    def __init__(self, config, intent_label_lst, slot_label_lst):
        # multi_intent: 1,
        # intent_seq: 1,
        # tag_intent: 1,
        # bi_tag: 1,
        # cls_token_cat: 1,
        # intent_attn: 1,
        # num_mask: 4
        super(MobileBertModel, self).__init__(config)
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.mobilebert = MobileBertModel(config=config)
        self.config = config
        self.multi_intent_classifier = MultiIntentClassifier(config.hidden_size, self.num_intent_labels, 0.1)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, 0.1)
        self.intent_token_classifier = IntentTokenClassifier(config.hidden_size, self.num_intent_labels, 0.1)

        self.pro_classifier = ProClassifier(2 * config.hidden_size, 0.1)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                pro_labels_ids): 
        
        outputs = self.mobilebert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # last-layer hidden-state, (hidden_states), (attentions)
        sequence_output = outputs[0]

        # ==================================== 2. Slot Softmax ========================================
        # (batch_size, seq_len, num_slots)
        slot_logits = self.slot_classifier(sequence_output)

        if 1 in pro_labels_ids:  # if use pro, and the batch contain pronouns
            # 1. concate pronoun to each word in the sequence
            pro_token_mask = pro_labels_ids > 0
            pro_sample_mask = torch.max(pro_token_mask.long(),dim = 1)[0] > 0
            pro_vec = sequence_output[pro_token_mask]
            pro_sequence_output = sequence_output[pro_sample_mask]
            pro_vec = pro_vec[:, None, :]  # add new dimention
            repeat_pro = pro_vec.repeat(1, 32, 1)#self.args.max_seq_len
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
        else:
            referee_token_logits = None
            full_referee_token_logits = None

        # ==================================== 3. Intent Token Softmax ========================================
        # if self.args.intent_seq:
            # (batch_size, seq_len, num_intents)
        intent_token_logits = self.intent_token_classifier(sequence_output)

        return slot_logits, intent_token_logits, referee_token_logits,full_referee_token_logits
    