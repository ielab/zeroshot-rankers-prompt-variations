from typing import List, Tuple
from .rankers import LlmRanker, SearchResult
import openai
import time
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import copy
from collections import Counter
import tiktoken
import random

random.seed(929)
import os

os.environ['CURL_CA_BUNDLE'] = ''
import certifi
import urllib3

http = urllib3.PoolManager(
    cert_reqs='CERT_REQUIRED',
    ca_certs=certifi.where()
)


def prompt_generator_t5(prompt_type, original_prompt_number, instruction, output, tone, order, position, role, query,
                        passages):
    if prompt_type == "original":
        original_prompt_dict = {
            1: f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
               + passages + '\n\n  Output only the passage label of the most relevant passage:',
        }
        return original_prompt_dict[original_prompt_number]

    elif prompt_type == "adjusted":
        instruction_dict = {
            "instruction_1": 'Given a query "{query}", which of the following passages is the most relevant one to the query?\n',
        }

        output_dict = {
            "output_1": "Output the passage label of the most relevant passage. ",
            "output_2": "Generate the passage label. ",
            "output_3": "Generate the passage label that is the most relevant to the query, then explain why you "
                        "think this passage is the most relevant. ",
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

        prompt_instruction = instruction_dict[instruction].format(query=query)
        passages = "\n" + passages + "\n"

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


def prompt_generator_mistral(prompt_type, original_prompt_number, instruction, output, tone, order, position, role,
                             query,
                             passages):
    if prompt_type == "original":
        original_prompt_dict = {
            1: f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
               + passages + '\n\n  Output only the passage label of the most relevant passage:',
        }

        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"

        role_text = ""
        sys_format = prefix + "{system}\n" + role_text + suffix
        prompt = prefix + "{user}\n" + original_prompt_dict[original_prompt_number] + suffix
        prompt_package = "<s>[INST] " + sys_format + prompt + "[/INST]</s>"

        return prompt_package

    elif prompt_type == "adjusted":
        instruction_dict = {
            "instruction_1": 'Given a query "{query}", which of the following passages is the most relevant one to the query?\n',
        }

        output_dict = {
            "output_1": "Output the passage label of the most relevant passage. ",
            "output_2": "Generate the passage label. ",
            "output_3": "Generate the passage label that is the most relevant to the query, then explain why you "
                        "think this passage is the most relevant. ",
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

        prompt_instruction = instruction_dict[instruction].format(query=query)
        passages = "\n" + passages + "\n"

        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"

        sys_format = prefix + "{system}\n" + role_dict[role] + suffix

        if order == "query_first":
            if position == "beginning":
                prompt = prefix + "{user}\n" + prompt_instruction + passages + tone_dict[tone] + output_dict[
                    output] + suffix
            elif position == "ending":
                prompt = prefix + "{user}\n" + tone_dict[tone] + output_dict[
                    output] + prompt_instruction + passages + suffix
            else:
                raise NotImplementedError(f"Position {position} is not implemented.")
        elif order == "passage_first":
            if position == "beginning":
                prompt = prefix + "{user}\n" + passages + prompt_instruction + tone_dict[tone] + output_dict[
                    output] + suffix
            elif position == "ending":
                prompt = prefix + "{user}\n" + tone_dict[tone] + output_dict[
                    output] + passages + prompt_instruction + suffix
            else:
                raise NotImplementedError(f"Position {position} is not implemented.")
        else:
            raise NotImplementedError(f"Order {order} is not implemented, position is {position}.")

        prompt_package = "<s>[INST] " + sys_format + prompt + "[/INST]</s>"

        # conversation = [{"role": "user", "content": prompt_package}]

        return prompt_package


class SetwiseLlmRanker(LlmRanker):
    CHARACTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                  "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
                  "W"]  # "Passage X" and "Passage Y" will be tokenized into 3 tokens, so we dont use for now

    def __init__(self,
                 model_name_or_path,
                 tokenizer_name_or_path,
                 device,
                 num_child=3,
                 k=10,
                 scoring='generation',
                 method="heapsort",
                 num_permutation=1,
                 cache_dir=None,
                 hf_token=None,
                 prompt_type="adjusted",
                 original_prompt_number=1,
                 instruction="instruction_1",
                 output="output_1",
                 tone="tone_1",
                 order="query_first",
                 position="beginning",
                 role="False"):

        self.device = device

        # used for setwise configuration
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k

        # used for prompts
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
                                                         model_name_or_path,
                                                         cache_dir=cache_dir)
            self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                  device_map='auto',
                                                                  torch_dtype=torch.float16 if device == 'cuda'
                                                                  else torch.float32,
                                                                  cache_dir=cache_dir)
            self.decoder_input_ids = self.tokenizer.encode("<pad> Passage",
                                                           return_tensors="pt",
                                                           add_special_tokens=False).to(
                self.device) if self.tokenizer else None

            test = []
            for i in range(len(self.CHARACTERS)):
                test.append(f'<pad> Passage {self.CHARACTERS[i]}')

            self.target_token_ids = self.tokenizer.batch_encode_plus([f'<pad> Passage {self.CHARACTERS[i]}'
                                                                      for i in range(len(self.CHARACTERS))],
                                                                     return_tensors="pt",
                                                                     add_special_tokens=False,
                                                                     padding=True).input_ids[:, -1]
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

        self.scoring = scoring
        self.method = method

    def compare(self, query: str, docs: List):

        passages = "\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])

        if self.scoring == 'generation':
            if self.config.model_type == 't5':

                input_text = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
                                                 self.output, self.tone, self.order, self.position, self.role, query,
                                                 passages)

                if self.num_permutation == 1:

                    if not self.has_printed_input_text:
                        print("prompt is \n" + input_text + '\n')
                        # Set the flag to True after printing, to prevent future prints
                        self.has_printed_input_text = True

                    input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

                    output_ids = self.llm.generate(input_ids,
                                                   decoder_input_ids=self.decoder_input_ids,
                                                   max_new_tokens=2)[0]

                    output = self.tokenizer.decode(output_ids,
                                                   skip_special_tokens=True).strip()
                    # print("output is :\n" + output)
                    output = output[-1]

                else:
                    id_passage = [(i, p) for i, p in enumerate(docs)]
                    labels = [self.CHARACTERS[i] for i in range(len(docs))]
                    batch_data = []
                    for _ in range(self.num_permutation):
                        batch_data.append([random.sample(id_passage, len(id_passage)),
                                           random.sample(labels, len(labels))])

                    batch_ref = []
                    # input_text = []
                    for batch in batch_data:
                        ref = []
                        passages = []
                        characters = []
                        for p, c in zip(batch[0], batch[1]):
                            ref.append(p[0])
                            passages.append(p[1].text)
                            characters.append(c)
                        batch_ref.append((ref, characters))
                        passages = "\n".join(
                            [f'Passage {characters[i]}: "{passages[i]}"' for i in range(len(passages))])

                    # input_text.append(prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction, self.output, self.tone, self.order, self.position, self.role, query, passages))
                    # if self.num_permutation == 2:
                    #     print("prompt is \n" + input_text + '\n')
                    input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                    output_ids = self.llm.generate(input_ids,
                                                   decoder_input_ids=self.decoder_input_ids.repeat(input_ids.shape[0],
                                                                                                   1), max_new_tokens=2)
                    output = self.tokenizer.batch_decode(output_ids[:, self.decoder_input_ids.shape[1]:],
                                                         skip_special_tokens=True)

                    # print("output is :\n" + output)

                    # vote
                    candidates = []
                    for ref, result in zip(batch_ref, output):
                        result = result.strip().upper()
                        docids, characters = ref
                        if len(result) != 1 or result not in characters:
                            print(f"Unexpected output: {result}")
                            continue
                        win_doc = docids[characters.index(result)]
                        candidates.append(win_doc)

                    if len(candidates) == 0:
                        print(f"Unexpected voting: {output}")
                        output = "Unexpected voting."
                    else:
                        # handle tie
                        candidate_counts = Counter(candidates)
                        max_count = max(candidate_counts.values())
                        most_common_candidates = [candidate for candidate, count in candidate_counts.items() if
                                                  count == max_count]
                        if len(most_common_candidates) == 1:
                            output = self.CHARACTERS[most_common_candidates[0]]
                        else:
                            output = self.CHARACTERS[random.choice(most_common_candidates)]

            elif self.config.model_type == 'mistral' or self.config.model_type == 'llama' or self.config.model_type == 'gemma':

                input_text = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
                                                 self.output, self.tone, self.order, self.position, self.role,
                                                 query,
                                                 passages)

                conversation = [{"role": "user", "content": input_text}]

                prompt_tem = self.tokenizer.apply_chat_template(conversation, tokenize=False,
                                                                add_generation_prompt=True)

                prompt_tem += "Passage"  # must put this at the end of prompt in setwise

                if not self.has_printed_input_text:
                    print("prompt is \n" + prompt_tem + '\n')
                    # Set the flag to True after printing, to prevent future prints
                    self.has_printed_input_text = True

                # print("prompt is \n" + prompt_tem + '\n')

                input_ids = self.tokenizer(prompt_tem, return_tensors="pt", padding='longest').input_ids.to(self.device)

                output_ids = self.llm.generate(input_ids,
                                               do_sample=False,
                                               temperature=0.0,
                                               top_p=None,
                                               max_new_tokens=10)[0]
                # 10 tokens can give some flexibility, later will check output before return

                output = self.tokenizer.decode(output_ids[input_ids.shape[1]:],
                                               skip_special_tokens=True).strip().upper()

                # print(f"output is\n{output} \n")

            # elif self.config.model_type == 'gemma':
            #
            #     input_text = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
            #                                      self.output, self.tone, self.order, self.position, self.role,
            #                                      query,
            #                                      passages)
            #
            #     conversation = [{"role": "user", "content": input_text}]
            #
            #     prompt_tem = self.tokenizer.apply_chat_template(conversation, tokenize=False,
            #                                                     add_generation_prompt=True)
            #
            #     prompt_tem += "Passage"  # must put this at the end of prompt in setwise
            #
            #     if not self.has_printed_input_text:
            #         print("prompt is \n" + prompt_tem + '\n')
            #         # Set the flag to True after printing, to prevent future prints
            #         self.has_printed_input_text = True
            #
            #     # print("prompt is \n" + prompt_tem + '\n')
            #
            #     input_ids = self.tokenizer(prompt_tem, return_tensors="pt").input_ids.to(self.device)
            #
            #     output_ids = self.llm.generate(input_ids,
            #                                    do_sample=False,
            #                                    temperature=0.0,
            #                                    top_p=None,
            #                                    max_new_tokens=10)[0]
            #     # 10 tokens can give some flexibility, later will check output before return
            #
            #     output = self.tokenizer.decode(output_ids[input_ids.shape[1]:],
            #                                    skip_special_tokens=True).strip().upper()

            else:
                raise NotImplementedError

        elif self.scoring == 'likelihood':
            if self.config.model_type == 't5':
                input_text = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
                                                 self.output, self.tone, self.order, self.position, self.role, query,
                                                 passages)

                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                with torch.no_grad():
                    logits = self.llm(input_ids=input_ids, decoder_input_ids=self.decoder_input_ids).logits[0][-1]
                    distributions = torch.softmax(logits, dim=0)
                    scores = distributions[self.target_token_ids[:len(docs)]]
                    ranked = sorted(zip(self.CHARACTERS[:len(docs)], scores), key=lambda x: x[1], reverse=True)
                    output = ranked[0][0]

            else:
                raise NotImplementedError

        if len(output) == 1 and output in self.CHARACTERS:
            pass
        else:
            output_tem = output.split()
            try:
                if output_tem[0] == "PASSAGE":
                    output = output_tem[1]
                elif output_tem[0] in self.CHARACTERS:
                    output = output_tem[0]
                elif output_tem[0].replace(':', '') in self.CHARACTERS:
                    # Remove the colon (if present) and assign to output
                    output = output_tem[0].replace(':', '')
                elif output_tem[0].replace(',', '') in self.CHARACTERS:
                    # Remove the colon (if present) and assign to output
                    output = output_tem[0].replace(',', '')
                elif output_tem[0].replace('.', '') in self.CHARACTERS:
                    # Remove the colon (if present) and assign to output
                    output = output_tem[0].replace('.', '')
                elif output_tem[0].replace(';', '') in self.CHARACTERS:
                    # Remove the colon (if present) and assign to output
                    output = output_tem[0].replace(';', '')
                else:
                    output = output_tem[0]
                    print(f"Unexpected output: {output_tem[0]}")
            except:
                print(f"Unexpected output: {output}")

        # print(f"output is {output}")

        return output

    def heapify(self, arr, n, i, query):
        # Find largest among root and children
        if self.num_child * i + 1 < n:  # if there are children
            docs = [arr[i]] + arr[self.num_child * i + 1: min((self.num_child * (i + 1) + 1), n)]
            inds = [i] + list(range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n)))
            output = self.compare(query, docs)
            try:
                best_ind = self.CHARACTERS.index(output)
            except ValueError:
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            # If root is not largest, swap with largest and continue heapifying
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query)

    def heapSort(self, arr, query, k):
        n = len(arr)
        ranked = 0
        # Build max heap
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query)
        for i in range(n - 1, 0, -1):
            # Swap
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            # Heapify root element
            self.heapify(arr, i, 0, query)

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)

        if self.method == "heapsort":
            self.heapSort(ranking, query, self.k)
            ranking = list(reversed(ranking))

        # elif self.method == "bubblesort":
        #     for i in range(k):
        #         start_ind = len(ranking) - (self.num_child + 1)
        #         end_ind = len(ranking)
        #         while True:
        #             if start_ind < i:
        #                 start_ind = i
        #             output = self.compare(query, ranking[start_ind:end_ind])
        #             try:
        #                 best_ind = self.CHARACTERS.index(output)
        #             except ValueError:
        #                 best_ind = 0
        #             if best_ind != 0:
        #                 ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
        #
        #             if start_ind == i:
        #                 break
        #
        #             start_ind -= self.num_child
        #             end_ind -= self.num_child
        elif self.method == "bubblesort":
            last_start = len(ranking) - (self.num_child + 1)

            for i in range(self.k):
                start_ind = last_start
                end_ind = last_start + (self.num_child + 1)
                is_change = False
                while True:
                    if start_ind < i:
                        start_ind = i
                    output = self.compare(query, ranking[start_ind:end_ind])
                    try:
                        best_ind = self.CHARACTERS.index(output)
                    except ValueError:
                        best_ind = 0
                    if best_ind != 0:
                        ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[
                            start_ind]
                        if not is_change:
                            is_change = True
                            if last_start != len(ranking) - (self.num_child + 1) \
                                    and best_ind == len(ranking[start_ind:end_ind]) - 1:
                                last_start += len(ranking[start_ind:end_ind]) - 1

                    if start_ind == i:
                        break

                    if not is_change:
                        last_start -= self.num_child

                    start_ind -= self.num_child
                    end_ind -= self.num_child

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


class OpenAiSetwiseLlmRanker(SetwiseLlmRanker):
    def __init__(self, model_name_or_path, api_key, num_child=3, method='heapsort', k=10):
        self.llm = model_name_or_path
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.num_child = num_child
        self.method = method
        self.k = k
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.system_prompt = "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pool of passages based on their relevance to the query."
        openai.api_key = api_key

    def compare(self, query: str, docs: List):
        self.total_compare += 1
        passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])
        input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                     + passages + '\n\nOutput only the passage label of the most relevant passage.'

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
                matches = re.findall(r"(Passage [A-Z])", output, re.MULTILINE)
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

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])
