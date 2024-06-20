from typing import List, Tuple
from .rankers import LlmRanker, SearchResult
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm
import copy
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
import tiktoken
import openai
import time
import re


def prompt_generator_t5(prompt_type, original_prompt_number, instruction, output, tone, order, position, role):
    if prompt_type == "original":
        original_prompt_dict = {
            1: "Given a query: {query}, which of the following two passages is more relevant to the query? \n"
               "Passage A: {doc1} \n"
               "Passage B: {doc2} \n"
               "Output Passage A or Passage B:"
        }
        return original_prompt_dict[original_prompt_number]

    elif prompt_type == "adjusted":
        instruction_dict = {
            "instruction_1": 'Given a query "{query}", '
                             'which of the following two passages is more relevant to the query?\n',
        }

        output_dict = {
            "output_1": "Output Passage A or Passage B. "
        }

        tone_dict = {
            "tone_0": "",
            "tone_1": "Please ",
            "tone_2": "Only ",
            "tone_3": "Must ",
            "tone_4": "You better get this right or you will be punished. ",
            "tone_5": "Only response the ranking results, do not say any word or explain. "
        }

        role_dict = {
            "True": "You are RankGPT, "
                    "an intelligent assistant that can rank passages based on their relevancy to the query. \n",
            "False": ""
        }

        prompt_instruction = instruction_dict[instruction]
        passages = "\nPassage A: {doc1} \nPassage B: {doc2} \n"

        if order == "query_first":
            if position == "beginning":
                prompt = role_dict[role] + prompt_instruction + passages + tone_dict[tone] + output_dict[output]
            elif position == "ending":
                prompt = role_dict[role] + tone_dict[tone] + output_dict[output] + prompt_instruction + passages
            else:
                raise NotImplementedError(f"Position {position} is not implemented.")
        elif order == "passage_first":
            if position == "beginning":
                prompt = role_dict[role] + passages + prompt_instruction + tone_dict[tone] + output_dict[output]
            elif position == "ending":
                prompt = role_dict[role] + tone_dict[tone] + output_dict[output] + passages + prompt_instruction
            else:
                raise NotImplementedError(f"Position {position} is not implemented.")
        else:
            raise NotImplementedError(f"Order {order} is not implemented, position is {position}.")

        return prompt


def prompt_generator_mistral(prompt_type, original_prompt_number, instruction, output, tone, order, position, role):
    if prompt_type == "original":
        original_prompt_dict = {
            1: "Given a query: {query}, which of the following two passages is more relevant to the query? \n "
               "Passage A: {doc1} \n"
               "Passage B: {doc2} \n"
               "Output Passage A or Passage B:"
        }

        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"

        role_text = ""
        sys_format = prefix + "{{system}}\n" + role_text + suffix
        prompt = prefix + "{{user}}\n" + original_prompt_dict[original_prompt_number] + suffix
        prompt_package = "<s>[INST] " + sys_format + prompt + "[/INST]</s>"

        return prompt_package

    elif prompt_type == "adjusted":
        instruction_dict = {
            "instruction_1": 'Given a query "{query}", '
                             'which of the following two passages is more relevant to the query?\n',
        }

        output_dict = {
            "output_1": "Output Passage A or Passage B. "
        }

        tone_dict = {
            "tone_0": "",
            "tone_1": "Please ",
            "tone_2": "Only ",
            "tone_3": "Must ",
            "tone_4": "You better get this right or you will be punished. ",
            "tone_5": "Only response the ranking results, do not say any word or explain. "
        }

        role_dict = {
            "True": "You are RankGPT, "
                    "an intelligent assistant that can rank passages based on their relevancy to the query. \n",
            "False": ""
        }

        prompt_instruction = instruction_dict[instruction]
        passages = "\nPassage A: {doc1} \nPassage B: {doc2} \n"

        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"

        sys_format = prefix + "{{system}}\n" + role_dict[role] + suffix

        if order == "query_first":
            if position == "beginning":
                prompt = prefix + "{{user}}\n" + prompt_instruction + passages + tone_dict[tone] + output_dict[
                    output] + suffix
            elif position == "ending":
                prompt = prefix + "{{user}}\n" + tone_dict[tone] + output_dict[
                    output] + prompt_instruction + passages + suffix
            else:
                raise NotImplementedError(f"Position {position} is not implemented.")
        elif order == "passage_first":
            if position == "beginning":
                prompt = prefix + "{{user}}\n" + passages + prompt_instruction + tone_dict[tone] + output_dict[
                    output] + suffix
            elif position == "ending":
                prompt = prefix + "{{user}}\n" + tone_dict[tone] + output_dict[
                    output] + passages + prompt_instruction + suffix
            else:
                raise NotImplementedError(f"Position {position} is not implemented.")
        else:
            raise NotImplementedError(f"Order {order} is not implemented, position is {position}.")

        prompt_package = "<s>[INST] " + sys_format + prompt + "[/INST]</s>"

        # conversation = [{"role": "user", "content": prompt_package}]

        return prompt_package


class Text2TextGenerationDataset(Dataset):
    def __init__(self, data: List[str], tokenizer):
        self.data = tokenizer(data)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, item):
        return {'input_ids': self.data['input_ids'][item],
                'attention_mask': self.data['attention_mask'][item]}


class PairwiseLlmRanker(LlmRanker):
    def __init__(self, model_name_or_path,
                 tokenizer_name_or_path,
                 device,
                 method="allpair",
                 batch_size=2,
                 k=10,
                 cache_dir=None,
                 hf_token=None,
                 prompt_type="adjusted",
                 original_prompt_number=1,
                 instruction="instruction_1",
                 output="output_1",
                 tone="tone_1",
                 order="query_first",
                 position="beginning",
                 role="False"
                 ):
        self.device = device
        self.method = method
        self.batch_size = batch_size
        self.k = k
        self.prompt_type = prompt_type
        self.original_prompt_number = original_prompt_number
        self.instruction = instruction
        self.output = output
        self.tone = tone
        self.order = order
        self.position = position
        self.role = role
        self.has_printed_input_text = False

        self.HF_TOKEN = hf_token
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir, token=self.HF_TOKEN)

        if self.config.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path
                                                         if tokenizer_name_or_path is not None else
                                                         model_name_or_path, cache_dir=cache_dir)
            self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                  device_map='auto',
                                                                  torch_dtype=torch.float16 if device == 'cuda'
                                                                  else torch.float32,
                                                                  cache_dir=cache_dir)
            self.decoder_input_ids = self.tokenizer.encode("<pad> Passage",
                                                           return_tensors="pt",
                                                           add_special_tokens=False).to(self.llm.device)
            self.decoder_input_ids = self.decoder_input_ids.repeat(self.batch_size, 1)

        elif self.config.model_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                           token=self.HF_TOKEN,
                                                           cache_dir=cache_dir)
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            token=self.HF_TOKEN,
                                                            device_map='auto',
                                                            torch_dtype=torch.bfloat16,
                                                            cache_dir=cache_dir)
            self.tokenizer.use_default_system_prompt = False
            self.tokenizer.pad_token = self.tokenizer.eos_token

        elif self.config.model_type == 'mistral':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            device_map='auto',
                                                            torch_dtype=torch.float16 if device == 'cuda'
                                                            else torch.float32,
                                                            cache_dir=cache_dir)
            self.tokenizer.use_default_system_prompt = False
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.padding_side = "left"
            self.llm.padding_side = "left"

        elif self.config.model_type == 'gemma':

            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                           token=self.HF_TOKEN,
                                                           cache_dir=cache_dir)
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            token=self.HF_TOKEN,
                                                            device_map='auto',
                                                            torch_dtype=torch.float16 if device == 'cuda'
                                                            else torch.float32,
                                                            revision="float16",
                                                            cache_dir=cache_dir)

        else:
            raise NotImplementedError

    def compare(self, query: str, docs: List):

        if self.config.model_type == 't5':

            prompt = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
                                         self.output, self.tone, self.order, self.position, self.role)

            doc1, doc2 = docs[0], docs[1]
            input_texts = [prompt.format(query=query, doc1=doc1, doc2=doc2),
                           prompt.format(query=query, doc1=doc2, doc2=doc1)]

            if not self.has_printed_input_text:
                print("prompt is \n" + input_texts[0] + '\n')
                # Set the flag to True after printing, to prevent future prints
                self.has_printed_input_text = True

            input_ids = self.tokenizer(input_texts,
                                       padding='longest',
                                       return_tensors="pt").input_ids.to(self.llm.device)

            output_ids = self.llm.generate(input_ids,
                                           decoder_input_ids=self.decoder_input_ids,
                                           max_new_tokens=2)

            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        elif self.config.model_type == 'llama' or self.config.model_type == 'mistral' or self.config.model_type == 'gemma':

            # prompt = prompt_generator_mistral(self.prompt_type, self.original_prompt_number, self.instruction,
            #                                   self.output, self.tone, self.order, self.position, self.role)
            # doc1, doc2 = docs[0], docs[1]
            # input_texts = [prompt.format(query=query, doc1=doc1, doc2=doc2),
            #                prompt.format(query=query, doc1=doc2, doc2=doc1)]

            prompt = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
                                         self.output, self.tone, self.order, self.position, self.role)

            doc1, doc2 = docs[0], docs[1]
            input_texts = [prompt.format(query=query, doc1=doc1, doc2=doc2),
                           prompt.format(query=query, doc1=doc2, doc2=doc1)]

            conversation0 = [{"role": "user", "content": input_texts[0]}]
            conversation1 = [{"role": "user", "content": input_texts[1]}]

            prompt0 = self.tokenizer.apply_chat_template(conversation0, tokenize=False, add_generation_prompt=True)
            prompt0 += " Passage"
            # print("prompt is \n" + prompt0 + '\n')
            prompt1 = self.tokenizer.apply_chat_template(conversation1, tokenize=False, add_generation_prompt=True)
            prompt1 += " Passage"

            # if not self.has_printed_input_text:
            #     print("prompt is \n" + prompt0 + '\n')
            #     # Set the flag to True after printing, to prevent future prints
            #     self.has_printed_input_text = True

            input_ids = self.tokenizer([prompt0, prompt1], return_tensors="pt", padding="longest").input_ids.to(
                self.device)

            output_ids = self.llm.generate(input_ids,
                                           do_sample=False,
                                           temperature=0.0,
                                           top_p=None,
                                           max_new_tokens=1)

            output0 = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:],
                                            skip_special_tokens=True).strip().upper()
            output1 = self.tokenizer.decode(output_ids[1][input_ids.shape[1]:],
                                            skip_special_tokens=True).strip().upper()

            # print(f"output is\nPassage {output0}\n", f"Passage {output1}\n")
            return [f'Passage {output0}', f'Passage {output1}']


        # elif self.config.model_type == 'gemma':
        #     prompt = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
        #                                  self.output, self.tone, self.order, self.position, self.role)
        #
        #     doc1, doc2 = docs[0], docs[1]
        #     input_texts = [prompt.format(query=query, doc1=doc1, doc2=doc2),
        #                    prompt.format(query=query, doc1=doc2, doc2=doc1)]
        #
        #     conversation0 = [{"role": "user", "content": input_texts[0]}]
        #     conversation1 = [{"role": "user", "content": input_texts[1]}]
        #
        #     prompt0 = self.tokenizer.apply_chat_template(conversation0, tokenize=False, add_generation_prompt=True)
        #     prompt0 += " Passage"
        #     print("prompt is \n" + prompt0 + '\n')
        #     prompt1 = self.tokenizer.apply_chat_template(conversation1, tokenize=False, add_generation_prompt=True)
        #     prompt1 += " Passage"
        #
        #     if not self.has_printed_input_text:
        #         print("prompt is \n" + prompt0 + '\n')
        #         # Set the flag to True after printing, to prevent future prints
        #         self.has_printed_input_text = True
        #
        #     input_ids = self.tokenizer([prompt0, prompt1], add_special_tokens=False,
        #                                return_tensors="pt").input_ids.to(self.device)
        #
        #     output_ids = self.llm.generate(input_ids,
        #                                    do_sample=False,
        #                                    temperature=0.0,
        #                                    top_p=None,
        #                                    max_new_tokens=1)
        #
        #     output0 = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:],
        #                                     skip_special_tokens=True).strip().upper()
        #     output1 = self.tokenizer.decode(output_ids[1][input_ids.shape[1]:],
        #                                     skip_special_tokens=True).strip().upper()
        #
        #     print(f"output is\nPassage {output0}\n", f"Passage {output1}\n")
        #     return [f'Passage {output0}', f'Passage {output1}']

        # elif self.config.model_type == 'mistral':

        #     prompt = prompt_generator_mistral(self.prompt_type, self.original_prompt_number, self.instruction,
        #                                       self.output, self.tone, self.order, self.position, self.role)

        #     prompt += " Passage label:"

        #     doc1, doc2 = docs[0], docs[1]
        #     input_texts = [prompt.format(query=query, doc1=doc1, doc2=doc2),
        #                    prompt.format(query=query, doc1=doc2, doc2=doc1)]

        #     if not self.has_printed_input_text:
        #         print("prompt is \n" + input_texts[0] + '\n')
        #         # Set the flag to True after printing, to prevent future prints
        #         self.has_printed_input_text = True

        #     input_ids = self.tokenizer([input_texts[0], input_texts[1]], return_tensors="pt").input_ids.to(self.device)

        #     output_ids = self.llm.generate(input_ids,
        #                                    do_sample=False,
        #                                    temperature=0.0,
        #                                    top_p=None,
        #                                    max_new_tokens=3)
        #     output0 = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:],
        #                                     skip_special_tokens=True).strip().upper().split()
        #     try:
        #         if output0[0] == 'A' or output0[0] == 'B':
        #             output0 = output0[0]
        #         elif output0[0] == "PASSAGE":
        #             output0 = output0[1]
        #         else:
        #             print(f'Unexpected output0: {output0}')
        #             output0 = "Unexpected"

        #         output1 = self.tokenizer.decode(output_ids[1][input_ids.shape[1]:],
        #                                         skip_special_tokens=True).strip().upper().split()

        #         if output1[0] == 'A' or output1[0] == 'B':
        #             output1 = output1[0]
        #         elif output1[0] == "PASSAGE":
        #             output1 = output1[1]
        #         else:
        #             print(f'Unexpected output1: {output1}')
        #             output1 = "Unexpected"
        #     except:
        #         output0 = "Unexpected"
        #         output1 = "Unexpected"

        #     print(f'Passage {output0}', f'Passage {output1}')

        #     return [f'Passage {output0}', f'Passage {output1}']

        else:
            raise NotImplementedError

        return output

    def heapify(self, arr, n, i):
        # Find largest among root and children
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[l] > arr[i]:
            largest = l

        if r < n and arr[r] > arr[largest]:
            largest = r

        # If root is not largest, swap with largest and continue heapifying
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, n, largest)

    def heapSort(self, arr, k):
        n = len(arr)
        ranked = 0
        # Build max heap
        for i in range(n // 2, -1, -1):
            self.heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            # Swap
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            # Heapify root element
            self.heapify(arr, i, 0)

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)

        # if self.method == "allpair":
        #     if self.config.model_type == 'mistral':
        #         doc_pairs = list(combinations(ranking, 2))
        #         allpairs = []
        #
        #         self.prompt = prompt_dict[str(self.num_prompt)]
        #         # self.prompt = original_prompt[str(self.num_prompt)]
        #
        #         sys_prompt = "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
        #
        #         prefix = "<|im_start|>"
        #         suffix = "<|im_end|>\n"
        #
        #         sys_format = prefix + "system\n" + sys_prompt + suffix
        #         user_format = prefix + "user\n" + self.prompt + suffix
        #
        #         for doc1, doc2 in tqdm(doc_pairs):
        #             if int(self.num_prompt) in range(1, 25):
        #                 input_texts = [user_format.format(query=query, doc1=doc1.text, doc2=doc2.text),
        #                                user_format.format(query=query, doc1=doc2.text, doc2=doc1.text)]
        #             else:
        #                 input_texts = [sys_format + user_format.format(query=query, doc1=doc1.text, doc2=doc2.text),
        #                                sys_format + user_format.format(query=query, doc1=doc2.text, doc2=doc1.text)]
        #
        #             conversation0 = [{"role": "user", "content": input_texts[0]}]
        #             conversation1 = [{"role": "user", "content": input_texts[1]}]
        #
        #             prompt0 = self.tokenizer.apply_chat_template(conversation0, tokenize=False,
        #                                                          add_generation_prompt=True)
        #             # prompt0 += " Passage:"
        #             prompt1 = self.tokenizer.apply_chat_template(conversation1, tokenize=False,
        #                                                          add_generation_prompt=True)
        #             # prompt1 += " Passage:"
        #
        #             allpairs.append(prompt0)
        #             allpairs.append(prompt1)
        #
        #
        #     else:
        #         doc_pairs = list(combinations(ranking, 2))
        #         allpairs = []
        #         for doc1, doc2 in tqdm(doc_pairs):
        #             allpairs.append(self.prompt.format(query=query, doc1=doc1.text, doc2=doc2.text))
        #             allpairs.append(self.prompt.format(query=query, doc1=doc2.text, doc2=doc1.text))
        #
        #     allpairs_dataset = Text2TextGenerationDataset(allpairs, self.tokenizer)
        #
        #     loader = DataLoader(
        #         allpairs_dataset,
        #         batch_size=self.batch_size,
        #         collate_fn=DataCollatorWithPadding(
        #             self.tokenizer,
        #             max_length=2048,
        #             padding='longest',
        #         ),
        #         shuffle=False,
        #         drop_last=False,
        #         num_workers=0
        #     )
        #
        #     outputs = []
        #     for batch_inputs in tqdm(loader):
        #         self.total_compare += 1
        #         self.total_prompt_tokens += batch_inputs['input_ids'].shape[0] * batch_inputs['input_ids'].shape[1]
        #
        #         if self.config.model_type == 't5':
        #             batch_outputs = self.llm.generate(batch_inputs['input_ids'].to(self.llm.device),
        #                                               decoder_input_ids=self.decoder_input_ids
        #                                               if self.decoder_input_ids.shape[0] == len(
        #                                                   batch_inputs['input_ids'])
        #                                               else self.decoder_input_ids[:len(batch_inputs['input_ids']), :],
        #                                               # last batch might be smaller
        #                                               max_new_tokens=2)
        #         else:
        #             batch_outputs = self.llm.generate(batch_inputs['input_ids'].to(self.llm.device),
        #                                               max_new_tokens=2)
        #
        #         self.total_completion_tokens += batch_outputs.shape[0] * batch_outputs.shape[1]
        #         outputs.extend(batch_outputs.cpu().numpy())
        #
        #         # print(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))
        #
        #     outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #
        #     scores = defaultdict(float)
        #     for i in range(0, len(outputs), 2):
        #         doc1, doc2 = doc_pairs[i // 2]
        #         output1 = outputs[i]
        #         output2 = outputs[i + 1]
        #
        #         if len(output1) > 9:
        #             output1 = output1[:9]
        #         if len(output2) > 9:
        #             output2 = output2[:9]
        #         # print("\n outputs is \n")
        #         # print(output1, output2)
        #
        #
        #         if output1 == "Passage A" and output2 == "Passage B":
        #             scores[doc1.docid] += 1
        #         elif output1 == "Passage B" and output2 == "Passage A":
        #             scores[doc2.docid] += 1
        #         else:  # conflict
        #             scores[doc1.docid] += 0.5
        #             scores[doc2.docid] += 0.5
        #
        #     ranking = sorted([SearchResult(docid=docid, score=score, text=None) for docid, score in scores.items()],
        #                      key=lambda x: x.score, reverse=True)

        if self.method == "heapsort":
            class ComparableDoc:
                def __init__(self, docid, text, ranker):
                    self.docid = docid
                    self.text = text
                    self.ranker = ranker

                def __gt__(self, other):
                    out = self.ranker.compare(query, [self.text, other.text])
                    if out[0] == "Passage A" and out[1] == "Passage B":
                        return True
                    else:
                        return False

            arr = [ComparableDoc(docid=doc.docid, text=doc.text, ranker=self) for doc in ranking]
            self.heapSort(arr, self.k)
            ranking = [SearchResult(docid=doc.docid, score=-i, text=None) for i, doc in enumerate(reversed(arr))]

        #
        # elif self.method == "bubblesort":
        #     k = min(k, len(ranking))
        #     for i in range(k):
        #         current_ind = len(ranking) - 1
        #         while True:
        #             if current_ind == i:
        #                 break
        #             doc1 = ranking[current_ind]
        #             doc2 = ranking[current_ind - 1]
        #             output = self.compare(query, [doc1.text, doc2.text])
        #             if output[0] == "Passage A" and output[1] == "Passage B":
        #                 ranking[current_ind - 1], ranking[current_ind] = ranking[current_ind], ranking[current_ind - 1]
        #             current_ind -= 1
        elif self.method == "bubblesort":
            k = min(self.k, len(ranking))

            last_end = len(ranking) - 1
            for i in range(k):
                current_ind = last_end
                is_change = False
                while True:
                    if current_ind <= i:
                        break
                    doc1 = ranking[current_ind]
                    doc2 = ranking[current_ind - 1]
                    output = self.compare(query, [doc1.text, doc2.text])
                    if output[0] == "Passage A" and output[1] == "Passage B":
                        ranking[current_ind - 1], ranking[current_ind] = ranking[current_ind], ranking[current_ind - 1]

                        if not is_change:
                            is_change = True
                            if last_end != len(ranking) - 1:  # skip unchanged pairs at the bottom
                                last_end += 1
                    if not is_change:
                        last_end -= 1
                    current_ind -= 1
        else:
            raise NotImplementedError(f'Method {self.method} is not implemented.')

        results = []
        top_doc_ids = set()
        rank = 1
        for i, doc in enumerate(ranking[:self.k]):
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1
        return results

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])


class OpenAiPairwiseLlmRanker(PairwiseLlmRanker):
    def __init__(self,
                 model_name_or_path,
                 api_key,
                 method="heapsort",
                 batch_size=2,
                 k=10):
        self.llm = model_name_or_path
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.method = method
        self.k = k
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.CHARACTERS = ["A", "B"]
        self.system_prompt = "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pair of passages based on their relevance to the query."
        self.prompt = """Given a query "{query}", which of the following two passages is more relevant to the query?
        
Passage A: "{doc1}"

Passage B: "{doc2}"

Output Passage A or Passage B:"""
        openai.api_key = api_key

    def _get_response(self, input_text):
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.llm,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_text},
                    ],
                    temperature=0.0,
                    request_timeout=15
                )
                self.total_completion_tokens += int(response['usage']['completion_tokens'])
                self.total_prompt_tokens += int(response['usage']['prompt_tokens'])

                output = response['choices'][0]['message']['content']
                matches = re.findall(r"(Passage [A-B])", output, re.MULTILINE)
                if matches:
                    output = matches[0][8]
                elif output.strip() in self.CHARACTERS:
                    pass
                else:
                    print(f"Unexpected output: {output}")
                    output = "A"
                return output

            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                time.sleep(5)
                continue
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
                time.sleep(5)
                continue
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                print(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(5)
                continue
            except openai.error.InvalidRequestError as e:
                # Handle invalid request error
                print(f"OpenAI API request was invalid: {e}")
                raise e
            except openai.error.AuthenticationError as e:
                # Handle authentication error
                print(f"OpenAI API request failed authentication: {e}")
                raise e
            except openai.error.Timeout as e:
                # Handle timeout error
                print(f"OpenAI API request timed out: {e}")
                time.sleep(5)
                continue
            except openai.error.ServiceUnavailableError as e:
                # Handle service unavailable error
                print(f"OpenAI API request failed with a service unavailable error: {e}")
                time.sleep(5)
                continue
            except Exception as e:
                print(f"Unknown error: {e}")
                raise e

    def compare(self, query: str, docs: List):
        self.total_compare += 1
        doc1, doc2 = docs[0], docs[1]
        input_texts = [self.prompt.format(query=query, doc1=doc1, doc2=doc2),
                       self.prompt.format(query=query, doc1=doc2, doc2=doc1)]

        return [f'Passage {self._get_response(input_texts[0])}', f'Passage {self._get_response(input_texts[1])}']

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])
