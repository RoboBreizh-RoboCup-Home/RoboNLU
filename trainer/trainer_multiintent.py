import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from utils import MODEL_CLASSES, get_intent_labels, get_slot_labels,compute_metrics_final

FLAG = False
logger = logging.getLogger(__name__)
class Trainer_multi(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        
        self.slot_preds_list = None
        self.intent_token_preds_list = None
        
        
        # set of intents
        self.intent_label_lst = get_intent_labels(args)
        # set of slots
        self.slot_label_lst = get_slot_labels(args)

        self.num_intent_labels = len(self.intent_label_lst)
        self.num_slot_labels = len(self.slot_label_lst)

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        # self.config = BertConfig.from_pretrained('bert-base-uncased', finetuning_task='gpsr_pro_instance')

        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      intent_label_lst=self.intent_label_lst,
                                                      slot_label_lst=self.slot_label_lst,
                                                      )

        for name, param in self.model.named_parameters():
            if name.startswith("encoder.layer.11") or name.startswith("encoder.layer.10"): # unfroze last 2 layers
                param.requires_grad = True


        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def get_intent_token_loss(self,intent_token_logits,intent_token_ids,attention_mask):
        intent_token_loss = 0.0
        intent_token_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
        if attention_mask is not None:
            active_intent_loss = attention_mask.view(-1) == 1
            active_intent_logits = intent_token_logits.view(-1, self.num_intent_labels)[active_intent_loss]
            active_intent_tokens = intent_token_ids.view(-1)[active_intent_loss]
            intent_token_loss = intent_token_loss_fct(active_intent_logits, active_intent_tokens)
        else:
            intent_token_loss = intent_token_loss_fct(intent_token_logits.view(-1, self.num_intent_labels), intent_token_ids.view(-1))

        return self.args.slot_loss_coef * intent_token_loss

    def get_slot_loss(self,slot_logits,slot_labels_ids,attention_mask):
        slot_loss = 0.0
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
                logger.info('attention_mask: ', attention_mask_cpu)
                logger.info('active_loss: ', active_loss_cpu)
        else:
            slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))

        return self.args.slot_loss_coef * slot_loss

    def get_referee_token_loss(self,referee_token_logits,referee_labels_ids,attention_mask,pro_sample_mask):
        referee_token_loss = 0.0
        class_weights = torch.FloatTensor([1,10,200]).to(self.device)
        referee_token_loss_fct = nn.CrossEntropyLoss(weight = class_weights,ignore_index=self.args.ignore_index) #self.referee_token_loss_fct

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
                logger.info('attention_mask: ', attention_mask_cpu)
                logger.info('active_loss: ', active_loss_cpu)
        else:
            referee_token_loss = referee_token_loss_fct(referee_token_logits.view(-1, 3), referee_labels_ids[pro_sample_mask].view(-1))

        return self.args.pro_loss_coef * referee_token_loss

    def train(self):
        
        #logger.info(vars(self.args))
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        step_per_epoch = len(train_dataloader) // 2

        # record the evaluation loss
        eval_acc = 0.0
        MAX_RECORD = self.args.patience
        num_eval = -1
        eval_result_record = (num_eval, eval_acc)
        flag = False
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=FLAG)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                attention_mask = batch[1]
                intent_label_ids = batch[3]
                slot_labels_ids =  batch[4]
                intent_token_ids =  batch[5]
                B_tag_mask =  batch[6]
                BI_tag_mask =  batch[7]
                tag_intent_label =  batch[8]
                referee_labels_ids =  batch[9] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                pro_labels_ids = batch[10]

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'pro_labels_ids' : batch[10]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]


                outputs = self.model(**inputs)

                if self.args.pro and self.args.intent_seq:
                    slot_logits, intent_token_logits, referee_token_logits,all_referee_token_logits = outputs

                slot_loss = self.get_slot_loss(slot_logits,slot_labels_ids,attention_mask)
                intent_token_loss =  self.get_intent_token_loss(intent_token_logits,intent_token_ids,attention_mask)

                pro_token_mask = pro_labels_ids > 0
                pro_sample_mask = torch.max(pro_token_mask.long(),dim = 1)[0] > 0
                referee_token_loss = self.get_referee_token_loss(referee_token_logits,referee_labels_ids,attention_mask,pro_sample_mask)
                loss = slot_loss + intent_token_loss + referee_token_loss


                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    # if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    if self.args.logging_steps > 0 and global_step % step_per_epoch == 0:
                        logger.info("***** Training Step %d *****", step)
                        logger.info("  total_loss = %f", loss)
                        logger.info("  slot_loss = %f", slot_loss)
                        logger.info("  intent_token_loss = %f", intent_token_loss)
                        logger.info("  referee_token_loss = %f", referee_token_loss) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                        dev_result = self.evaluate("dev")
                        test_result = self.evaluate("test")
                        num_eval += 1
                        if self.args.patience != 0:
                            if dev_result['sementic_frame_acc'] + dev_result['intent_acc'] + dev_result['slot_f1']   > eval_result_record[1]:
                                self.save_model()
                                eval_result_record = (num_eval, dev_result['sementic_frame_acc'] + dev_result['intent_acc'] + dev_result['slot_f1'] )
                            else:
                                cur_num_eval = eval_result_record[0]
                                if num_eval - cur_num_eval >= MAX_RECORD:
                                    # it has been ok
                                    logger.info(' EARLY STOP Evaluate: at {}, best eval {} intent_slot_acc: {} '.format(num_eval, cur_num_eval, eval_result_record[1]))
                                    flag = True
                                    break
                        else:
                            self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if flag:
                train_iterator.close()
                break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        intent_token_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None
        out_intent_token_ids = None
        
        tag_intent_preds = None
        out_tag_intent_ids = None
        referee_preds = None #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        out_referee_labels_ids = None #!!!!!!!!!!!!!!!!!!!!!!!!!
        all_referee_preds = None #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        all_out_referee_labels_ids = None #!!!!!!!!!!!!!!!!!!!!!!!!!

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=FLAG):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                attention_mask = batch[1]
                intent_label_ids = batch[3]
                slot_labels_ids =  batch[4]
                intent_token_ids =  batch[5]
                B_tag_mask =  batch[6]
                BI_tag_mask =  batch[7]
                tag_intent_label =  batch[8]
                referee_labels_ids =  batch[9] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                pro_labels_ids = batch[10]

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'pro_labels_ids' : batch[10]}


                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)

            # logits = outputs[0]
            if self.args.pro and self.args.intent_seq:
                slot_logits, intent_token_logits, referee_token_logits,all_referee_token_logits = outputs

            slot_loss = self.get_slot_loss(slot_logits,slot_labels_ids,attention_mask)
            intent_token_loss =  self.get_intent_token_loss(intent_token_logits,intent_token_ids,attention_mask)

            pro_token_mask = pro_labels_ids > 0
            pro_sample_mask = torch.max(pro_token_mask.long(),dim = 1)[0] > 0
            referee_token_loss = self.get_referee_token_loss(referee_token_logits,referee_labels_ids,attention_mask,pro_sample_mask)
            loss = slot_loss + intent_token_loss + referee_token_loss

            # ============================= Slot prediction ==============================
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = slot_labels_ids.detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(out_slot_labels_ids, slot_labels_ids.detach().cpu().numpy(), axis=0)

            # ============================= Pronoun referee prediction ==============================
            if self.args.pro:
                if all_referee_preds is None:

                    all_referee_preds = all_referee_token_logits.detach().cpu().numpy()
                    referee_preds = referee_token_logits.detach().cpu().numpy()

                    pro_sample_mask_np = (torch.max(inputs["pro_labels_ids"],dim = 1)[0] > 0).detach().cpu().numpy()
                    all_out_referee_labels_ids = referee_labels_ids.detach().cpu().numpy()
                    out_referee_labels_ids = all_out_referee_labels_ids[pro_sample_mask_np]


                else:

                    all_referee_preds = np.append(all_referee_preds,all_referee_token_logits.detach().cpu().numpy(), axis = 0)
                    referee_preds = np.append(referee_preds, referee_token_logits.detach().cpu().numpy(), axis = 0)

                    pro_sample_mask_np = (torch.max(inputs["pro_labels_ids"],dim = 1)[0] > 0).detach().cpu().numpy()
                    new_all_out_referee_labels_ids = referee_labels_ids.detach().cpu().numpy()
                    all_out_referee_labels_ids = np.append(all_out_referee_labels_ids,new_all_out_referee_labels_ids,axis = 0)
                    new_out_referee_labels_ids = new_all_out_referee_labels_ids[pro_sample_mask_np]#np.array([ele for i,ele in enumerate(new_all_out_referee_labels_ids) if pro_sample_mask_np[i] != False])
                    out_referee_labels_ids = np.append(out_referee_labels_ids, new_out_referee_labels_ids, axis = 0)

            if self.args.intent_seq:
                if intent_token_preds is None:
                    if self.args.use_crf:
                        intent_token_preds = np.array(self.model.crf.decode(intent_token_logits))
                    else:
                        intent_token_preds = intent_token_logits.detach().cpu().numpy()

                    out_intent_token_ids = intent_token_ids.detach().cpu().numpy()
                else:
                    if self.args.use_crf:
                        intent_token_preds = np.append(intent_token_preds, np.array(self.model.crf.decode(intent_token_logits)), axis=0)
                    else:
                        intent_token_preds = np.append(intent_token_preds, intent_token_logits.detach().cpu().numpy(), axis=0)

                    out_intent_token_ids = np.append(out_intent_token_ids, intent_token_ids.detach().cpu().numpy(), axis=0)

            eval_loss += loss.item()
            nb_eval_steps += 1
            eval_loss = eval_loss / nb_eval_steps
            results = {
                "loss": eval_loss
            }
        nb_eval_steps += 1

        # Slot result
        # (batch_size, seq_len)
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        
        # generate mask
        for i in range(out_slot_labels_ids.shape[0]):
            # record the padding position
            pos_offset = [0 for _ in range(out_slot_labels_ids.shape[1])]
            pos_cnt = 0
            padding_recording = [0 for _ in range(out_slot_labels_ids.shape[1])]
            
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
                    pos_offset[pos_cnt+1] = pos_offset[pos_cnt]
                    pos_cnt += 1
                else:
                    pos_offset[pos_cnt] = pos_offset[pos_cnt] + 1
                    padding_recording[j] = 1

        referee_token_map = {0:'PAD', 1:'O' ,2: 'B-referee'} # All referee are just one word in EGPSR


        if self.args.pro:
            referee_preds = np.argmax(referee_preds, axis=2)
            all_referee_preds = np.argmax(all_referee_preds, axis=2)


            referee_preds_list = [[] for _ in range(out_referee_labels_ids.shape[0])]
            out_referee_label_list = [[] for _ in range(out_referee_labels_ids.shape[0])]
            all_referee_preds_list = [[] for _ in range(all_out_referee_labels_ids.shape[0])]
            all_out_referee_label_list = [[] for _ in range(all_out_referee_labels_ids.shape[0])]

            for i in range(out_referee_labels_ids.shape[0]):
                for j in range(out_referee_labels_ids.shape[1]):
                    if out_referee_labels_ids[i, j] != self.pad_token_label_id:
                        out_referee_label_list[i].append(referee_token_map[out_referee_labels_ids[i][j]])
                        referee_preds_list[i].append(referee_token_map[referee_preds[i][j]])

            for i in range(all_out_referee_labels_ids.shape[0]):
                for j in range(all_out_referee_labels_ids.shape[1]):
                    if all_out_referee_labels_ids[i, j] != self.pad_token_label_id:
                        all_out_referee_label_list[i].append(referee_token_map[all_out_referee_labels_ids[i][j]])
                        all_referee_preds_list[i].append(referee_token_map[all_referee_preds[i][j]])

        intent_token_map = {i: label for i, label in enumerate(self.intent_label_lst)}
        out_intent_token_list = None
        intent_token_preds_list = None
        # ============================= Intent Seq Prediction ============================
        if self.args.intent_seq:
            if not self.args.use_crf:
                intent_token_preds = np.argmax(intent_token_preds, axis=2)
            out_intent_token_list = [[] for _ in range(out_intent_token_ids.shape[0])]
            intent_token_preds_list = [[] for _ in range(out_intent_token_ids.shape[0])]

            for i in range(out_intent_token_ids.shape[0]):
                for j in range(out_intent_token_ids.shape[1]):
                    if out_intent_token_ids[i, j] != self.pad_token_label_id:
                        out_intent_token_list[i].append(intent_token_map[out_intent_token_ids[i][j]])
                        intent_token_preds_list[i].append(intent_token_map[intent_token_preds[i][j]])


        total_result = compute_metrics_final(
                                       slot_preds_list,
                                       out_slot_label_list,
                                       intent_token_preds_list,
                                       out_intent_token_list,
                                       referee_preds_list,
                                       out_referee_label_list
                                      )
        results.update(total_result)

        correct = 0
        for ref_pred_seq,ref_label_seq in zip(referee_preds_list,out_referee_label_list):
            if ref_pred_seq == ref_label_seq:
                correct += 1
        ref_acc = correct/len(referee_preds_list)
        print('Pronoun Accurac: ',ref_acc, ', correct: ',correct, ', total: ',len(referee_preds_list))
        results.update({'Pronoun Accuracy':ref_acc})


        correct = 0
        for slot_pred_seq,slot_label_seq in zip(slot_preds_list,out_slot_label_list):
            if slot_pred_seq == slot_label_seq:
                correct += 1

        slot_acc = correct/len(slot_preds_list)
        print('Slot Accurac: ',slot_acc, ', correct: ',correct, ', total: ',len(slot_preds_list))

        correct = 0
        for intent_pred_seq,intent_label_seq in zip(intent_token_preds_list,out_intent_token_list):
            if intent_pred_seq == intent_label_seq:
                correct += 1
        intent_acc = correct/len(intent_token_preds_list)
        print('Intent Accurac: ',intent_acc, ', correct: ',correct, ', total: ',len(intent_token_preds_list))

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s_%s = %s", mode, key, str(results[key]))
        
        #self.store_pred(slot_preds_list,intent_token_preds_list)
        self.slot_preds_list = slot_preds_list
        self.intent_token_preds_list = intent_token_preds_list
        return results
    
    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        logger.info("Saving model checkpoint to %s", self.args.model_dir)
        torch.save(self.model, self.args.model_dir+"/model.pth")


    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")