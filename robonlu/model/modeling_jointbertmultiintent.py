import torch
import torch.nn as nn
# from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
# from torchcrf import CRF
# from TorchCRF import CRF
from .module import IntentClassifier, SlotClassifier, IntentTokenClassifier, MultiIntentClassifier, TagIntentClassifier, ProClassifier
import logging

logger = logging.getLogger()



class JointBERTMultiIntent(BertPreTrainedModel):
    # multi_intent: 1,
    # intent_seq: 1,
    # tag_intent: 1,
    # bi_tag: 1,
    # cls_token_cat: 1,
    # intent_attn: 1,
    # num_mask: 4
    def __init__(self, config, intent_label_lst, slot_label_lst):
        super().__init__(config)
        # self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        # load pretrain bert
        self.bert = BertModel(config=config)
        self.config = config
        # self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.multi_intent_classifier = MultiIntentClassifier(config.hidden_size, self.num_intent_labels, 0.1)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, 0.1)
        # if args.intent_seq:
        self.intent_token_classifier = IntentTokenClassifier(config.hidden_size, self.num_intent_labels, 0.1)

        # if args.pro:
        self.pro_classifier = ProClassifier(2 * config.hidden_size, 0.1)

        # if args.tag_intent:
        #     if args.cls_token_cat:
        #         self.tag_intent_classifier = TagIntentClassifier(2 * config.hidden_size, self.num_intent_labels, args.dropout_rate)
        #     else:
        #         self.tag_intent_classifier = TagIntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        #
        # if args.use_crf:
        #     self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                pro_labels_ids):  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! adding in pro data

        """
            Args: B: batch_size; L: sequence length; I: the number of intents; M: number of mask; D: the output dim of Bert
            input_ids: B * L
            token_type_ids: B * L
            token_type_ids: B * L
            intent_label_ids: B * I
            slot_labels_ids: B * L
            intent_token_ids: B * L
            B_tag_mask: B * M * L
            BI_tag_mask: B * M * L
            tag_intent_label: B * M
        """
        # input_ids:  torch.Size([32, 50])
        # attention_mask:  torch.Size([32, 50])
        # token_type_ids:  torch.Size([32, 50])
        # intent_label_ids:  torch.Size([32, 10])
        # slot_labels_ids:  torch.Size([32, 50])
        # intent_token_ids:  torch.Size([32, 50])
        # B_tag_mask:  torch.Size([32, 4, 50])
        # BI_tag_mask:  torch.Size([32, 4, 50])
        # tag_intent_label:  torch.Size([32, 4])
        
        # (len_seq, batch_size, hidden_dim), (batch_size, hidden_dim)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        # B * L * D
        sequence_output = outputs[0]
        # B * D
        pooled_output = outputs[1]  # [CLS]
        
        total_loss = 0

            
        # ==================================== 2. Slot Softmax ========================================
        # (batch_size, seq_len, num_slots)
        slot_logits = self.slot_classifier(sequence_output)


        # ==================================== NEW. Pronoun ========================================
        referee_token_loss = 0.0
        # self.args.pro and
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

            # print('sequence_output size: ', sequence_output.size())
            # print('pro_labels_ids size: ', pro_labels_ids.size())
            # print('pro_token_mask size: ', pro_token_mask.size())
            # print('pro_sample_mask size: ', pro_sample_mask.size())
            # print('pro_vec size: ', pro_vec.size())
            # print('add new dim size: ', pro_vec.size())
            # print('after repeat size: ', repeat_pro.size())
            # print('concated_input size: ', concated_input.size())
            # print('referee_token_logits size: ',referee_token_logits.size())
            # print('slot_logits size: ', slot_logits.size())



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




        # if self.args.pro and self.args.intent_seq and self.args.tag_intent:
        #     mobile_bert_outputs = ((intent_logits, slot_logits, intent_token_logits, tag_intent_logits, referee_token_logits,full_referee_token_logits),) + mobile_bert_outputs[2:]




        # intent_logits = None
        # if self.args.pro and self.args.intent_seq:
        # mobile_bert_outputs = ((slot_logits, intent_token_logits, referee_token_logits,full_referee_token_logits),) + mobile_bert_outputs[2:]

        # elif self.args.intent_seq and self.args.tag_intent:
        #     mobile_bert_outputs = ((intent_logits, slot_logits, intent_token_logits, tag_intent_logits),) + mobile_bert_outputs[2:]  # add hidden states and attention if they are here
        # elif self.args.intent_seq:
        #     mobile_bert_outputs = ((intent_logits, slot_logits, intent_token_logits),) + mobile_bert_outputs[2:]
        # # elif self.args.tag_intent:
        # #     mobile_bert_outputs = ((intent_logits, slot_logits, tag_intent_logits),) + mobile_bert_outputs[2:]
        # else:
        #     mobile_bert_outputs = ((intent_logits, slot_logits),) + mobile_bert_outputs[2:]
        
        # mobile_bert_outputs = ([total_loss, intent_loss, slot_loss, intent_token_loss, tag_intent_loss,referee_token_loss],) + mobile_bert_outputs

        # print('len mobile_bert_outputs',len(mobile_bert_outputs))
        # print('len mobile_bert_outputs[:2][1]', len(mobile_bert_outputs[:2][1]))
        # print('len mobile_bert_outputs[:2][0]', len(mobile_bert_outputs[:2][0]))

        return slot_logits, intent_token_logits, referee_token_logits,full_referee_token_logits
        # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits