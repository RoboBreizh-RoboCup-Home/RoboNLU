import torch
import torch.nn as nn
# from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
# from torchcrf import CRF
from TorchCRF import CRF
from .module import IntentClassifier, SlotClassifier, IntentTokenClassifier, MultiIntentClassifier, TagIntentClassifier, ProClassifier
import logging

logger = logging.getLogger()



class JointBERTMultiIntent_old(BertPreTrainedModel):
    # multi_intent: 1,
    # intent_seq: 1,
    # tag_intent: 1,
    # bi_tag: 1,
    # cls_token_cat: 1,
    # intent_attn: 1,
    # num_mask: 4
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super().__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        # load pretrain bert
        self.bert = BertModel(config=config)
        self.config = config
        # self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.multi_intent_classifier = MultiIntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)
        if args.intent_seq:
            self.intent_token_classifier = IntentTokenClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)

        if args.pro:
            self.pro_classifier = ProClassifier(2 * config.hidden_size, args.dropout_rate)

        if args.tag_intent:
            if args.cls_token_cat:
                self.tag_intent_classifier = TagIntentClassifier(2 * config.hidden_size, self.num_intent_labels, args.dropout_rate)
            else:
                self.tag_intent_classifier = TagIntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        
        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                intent_label_ids,
                slot_labels_ids,
                intent_token_ids,
                B_tag_mask,
                BI_tag_mask,
                tag_intent_label,
                referee_labels_ids,
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
        
        # ==================================== 1. Intent Softmax ========================================
        # (batch_size, num_intents)
        intent_logits = self.multi_intent_classifier(pooled_output)
        intent_logits_cpu = intent_logits.data.cpu().numpy()
        
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1, self.num_intent_labels))
            else:
                # intent_loss_fct = nn.CrossEntropyLoss()
                # default reduction is mean
                intent_loss_fct = nn.BCELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels) + 1e-10, intent_label_ids.view(-1, self.num_intent_labels))
            # Question: do we need to add weight here
            total_loss += intent_loss
            
        if intent_label_ids.type() != torch.cuda.FloatTensor:
            intent_label_ids = intent_label_ids.type(torch.cuda.FloatTensor)
            
        # ==================================== 2. Slot Softmax ========================================
        # (batch_size, seq_len, num_slots)
        slot_logits = self.slot_classifier(sequence_output)
        
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    try:
                        active_loss = attention_mask.view(-1) == 1
                        attention_mask_cpu = attention_mask.data.cpu().numpy()
                        active_loss_cpu = active_loss.data.cpu().numpy()
                        active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                        active_labels = slot_labels_ids.view(-1)[active_loss]
                        slot_loss = slot_loss_fct(active_logits, active_labels)
                    except:
                        print('intent_logits: ', intent_logits_cpu)
                        print('attention_mask: ', attention_mask_cpu)
                        print('active_loss: ', active_loss_cpu)
                        logger.info('intent_logits: ', intent_logits_cpu)
                        logger.info('attention_mask: ', attention_mask_cpu)
                        logger.info('active_loss: ', active_loss_cpu)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))

            total_loss += self.args.slot_loss_coef * slot_loss

            # ==================================== NEW. Pronoun ========================================
            referee_token_loss = 0.0
            if self.args.pro and 1 in pro_labels_ids:  # if use pro, and the batch contain pronouns
                # 1. concate pronoun to each word in the sequence
                pro_token_mask = pro_labels_ids > 0
                pro_sample_mask = torch.max(pro_token_mask.long(),dim = 1)[0] > 0
                pro_vec = sequence_output[pro_token_mask]
                pro_sequence_output = sequence_output[pro_sample_mask]
                pro_vec = pro_vec[:, None, :]  # add new dimention
                repeat_pro = pro_vec.repeat(1, self.args.max_seq_len, 1)
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

                if referee_labels_ids is not None:
                    # referee_token_loss_fct = nn.BCELoss(ignore_index=self.args.ignore_index)

                    total_tokens = torch.numel(pro_token_mask)
                    pro_tokens = torch.sum(pro_token_mask)

                    # # and assume the weight of non-pro token as 1
                    # pro_weight = (total_tokens - pro_tokens)/pro_tokens
                    #
                    # flatten_token_map = torch.flatten(pro_token_mask)
                    # weights_lst = [1 if ele == False else pro_weight for ele in flatten_token_map]
                    #                     weights = torch.FloatTensor(weights_lst)

                    class_weights = torch.FloatTensor([1,10,200]).to('cuda')
                    referee_token_loss_fct = nn.CrossEntropyLoss(weight = class_weights,ignore_index=self.args.ignore_index) #self.referee_token_loss_fct
                    # referee_token_loss_fct = nn.BCELoss(weight = torch.FloatTensor([1,45])) # Should I give [PAD] and other special tokens a label, so that I could use ign_idx? and use attention mask. what are the benefits of att mask

                    if attention_mask is not None:
                        try:
                            active_loss = attention_mask[pro_sample_mask].view(-1) == 1
                            attention_mask_cpu = attention_mask.data.cpu().numpy()
                            active_loss_cpu = active_loss.data.cpu().numpy()
                            active_logits = referee_token_logits.view(-1, 3)[active_loss]
                            active_labels = referee_labels_ids[pro_sample_mask].view(-1)[active_loss]
                            referee_token_loss = referee_token_loss_fct(active_logits,
                                                                        active_labels)
                        except:
                            # print('referee_token_logits: ', referee_token_logits)
                            # print('referee_token_logits size: ',referee_token_logits.size())
                            # print('attention_mask: ', attention_mask_cpu)
                            # print('active_loss: ', active_loss_cpu)
                            logger.info('referee_token_logits: ', intent_logits_cpu)
                            logger.info('attention_mask: ', attention_mask_cpu)
                            logger.info('active_loss: ', active_loss_cpu)
                    else:
                        print('else')
                        referee_token_loss = referee_token_loss_fct(referee_token_logits.view(-1, 3), referee_labels_ids[pro_sample_mask].view(-1))



                    total_loss += self.args.pro_loss_coef * referee_token_loss

                    # for the prediction, we need the pro labels for all samples
                    full_referee_token_logits = torch.zeros((pro_labels_ids.shape[0], referee_token_logits.shape[1],referee_token_logits.shape[2]))
                    # full_referee_token_logits[pro_sample_mask] = referee_token_logits
                    counter = 0
                    for idx,sample_bool in enumerate(pro_sample_mask):
                        if sample_bool:
                            full_referee_token_logits[idx] = referee_token_logits[counter]
                            counter += 1

                                                                                                                # mask: only get the labels of the samples that has pros
                    # referee_token_loss = referee_token_loss_fct(referee_token_logits.view(-1),referee_labels_ids[pro_sample_mask].view(-1).to(torch.float32))  # !!!!!!!!!!!! view(-1) because it's binary classification
                    # total_loss += self.args.pro_loss_coef * referee_token_loss / (torch.sum(pro_sample_mask)) #divide by the number of pro samples


        
        # ==================================== 3. Intent Token Softmax ========================================
        intent_token_loss = 0.0
        if self.args.intent_seq:
            # (batch_size, seq_len, num_intents)
            intent_token_logits = self.intent_token_classifier(sequence_output)

            if intent_token_ids is not None:
                if self.args.use_crf:
                    intent_token_loss = self.crf(intent_token_logits, intent_token_ids, mask=attention_mask.byte, reduction='mean')
                    intent_token_loss = -1 * intent_token_loss
                else:
                    intent_token_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                    if attention_mask is not None:
                        active_intent_loss = attention_mask.view(-1) == 1
                        active_intent_logits = intent_token_logits.view(-1, self.num_intent_labels)[active_intent_loss]
                        active_intent_tokens = intent_token_ids.view(-1)[active_intent_loss]
                        intent_token_loss = intent_token_loss_fct(active_intent_logits, active_intent_tokens)
                    else:
                        intent_token_loss = intent_token_loss_fct(intent_token_logits.view(-1, self.num_intent_labels), intent_token_ids.view(-1))
                
                total_loss += self.args.slot_loss_coef * intent_token_loss
        
        # convert the sequence_out to long
        if BI_tag_mask != None and  BI_tag_mask.type() != torch.cuda.FloatTensor:
            BI_tag_mask = BI_tag_mask.type(torch.cuda.FloatTensor)

        if B_tag_mask != None and B_tag_mask.type() != torch.cuda.FloatTensor:
            B_tag_mask = B_tag_mask.type(torch.cuda.FloatTensor)
        
        tag_intent_loss = 0.0
        if self.args.tag_intent:
            # B * M * D
            if self.args.BI_tag:
                tag_intent_vec = torch.einsum('bml,bld->bmd', BI_tag_mask, sequence_output)
            else:
                tag_intent_vec = torch.einsum('bml,bld->bmd', B_tag_mask, sequence_output)
            
            if self.args.cls_token_cat:
                cls_token = pooled_output.unsqueeze(1)
                # B * M * D
                cls_token = cls_token.repeat(1, self.args.num_mask, 1)
                # B * M * 2D
                tag_intent_vec = torch.cat((cls_token, tag_intent_vec), dim=2)
            
            tag_intent_vec = tag_intent_vec.view(tag_intent_vec.size(0) * tag_intent_vec.size(1), -1)
            
            # after softmax
            tag_intent_logits = self.tag_intent_classifier(tag_intent_vec)

            if self.args.intent_attn:
                # (batch_size, num_intent) => (batch_size * num_mask, num_intent) sigmoid [0, 1]
                intent_probs = intent_logits.unsqueeze(1)
                intent_probs = intent_probs.repeat(1, self.args.num_mask, 1)
                intent_probs = intent_probs.view(intent_probs.size(0) * intent_probs.size(1), -1)
                # (batch_size * num_mask, num_intent)
                tag_intent_logits = tag_intent_logits * intent_probs
                tag_intent_logits = tag_intent_logits.div(tag_intent_logits.sum(dim=1, keepdim=True))
            
            # tag_intent_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)

            # tag_intent_loss = tag_intent_loss_fct(tag_intent_logits, tag_intent_label.view(-1))

            nll_fct = nn.NLLLoss(ignore_index=self.args.ignore_index)
            
            tag_intent_loss = nll_fct(torch.log(tag_intent_logits + 1e-10), tag_intent_label.view(-1))
            
            total_loss += self.args.tag_intent_coef * tag_intent_loss

        if self.args.pro and self.args.intent_seq and self.args.tag_intent:
            outputs = ((intent_logits, slot_logits, intent_token_logits, tag_intent_logits, referee_token_logits,full_referee_token_logits),) + outputs[2:]
            # print('pro')
            # print('mobile_bert_outputs[0] shape: ', len(mobile_bert_outputs[0]))
        if self.args.pro and self.args.intent_seq:
            outputs = ((intent_logits, slot_logits, intent_token_logits, referee_token_logits,full_referee_token_logits),) + outputs[2:]

        elif self.args.intent_seq and self.args.tag_intent:
            outputs = ((intent_logits, slot_logits, intent_token_logits, tag_intent_logits),) + outputs[2:]  # add hidden states and attention if they are here
        elif self.args.intent_seq:
            outputs = ((intent_logits, slot_logits, intent_token_logits),) + outputs[2:]
        elif self.args.tag_intent:
            outputs = ((intent_logits, slot_logits, tag_intent_logits),) + outputs[2:]
        else:
            outputs = ((intent_logits, slot_logits),) + outputs[2:]
        
        outputs = ([total_loss, intent_loss, slot_loss, intent_token_loss, tag_intent_loss,referee_token_loss],) + outputs

        # print('len mobile_bert_outputs',len(mobile_bert_outputs))
        # print('len mobile_bert_outputs[:2][1]', len(mobile_bert_outputs[:2][1]))
        # print('len mobile_bert_outputs[:2][0]', len(mobile_bert_outputs[:2][0]))

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits