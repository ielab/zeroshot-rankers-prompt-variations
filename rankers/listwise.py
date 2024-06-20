import tiktoken
from .rankers import LlmRanker, SearchResult
from typing import List, Tuple
import copy
import openai
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, AutoConfig


def max_tokens(model):
    if 'gpt-4' in model:
        return 8192
    else:
        return 4096


def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."


def get_prefix_prompt(query, num):
    return [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    if model == "gpt-3.5-turbo":
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        tokens_per_message, tokens_per_name = 0, 0

    try:
        encoding = tiktoken.get_encoding(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    if isinstance(messages, list):
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
    else:
        num_tokens += len(encoding.encode(messages))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def create_permutation_instruction_chat(query: str, docs: List[SearchResult], model_name='gpt-3.5-turbo'):
    num = len(docs)

    max_length = 300
    while True:
        messages = get_prefix_prompt(query, num)
        rank = 0
        for doc in docs:
            rank += 1
            content = doc.text
            content = content.replace('Title: Content: ', '')
            content = content.strip()
            # For Japanese should cut by character: content = content[:int(max_length)]
            content = ' '.join(content.split()[:int(max_length)])
            messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
            messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
        messages.append({'role': 'user', 'content': get_post_prompt(query, num)})

        if model_name is not None:
            if num_tokens_from_messages(messages, model_name) <= max_tokens(model_name) - 200:
                break
            else:
                max_length -= 1
        else:
            break
    return messages


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            try:
                int(c)
                new_response += c
            except:
                new_response += ' '
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(ranking, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    # response = [int(x) - 1 for x in response.split()]
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(ranking[rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]

    for j, x in enumerate(response):
        ranking[j + rank_start] = cut_range[x]

    return ranking


class OpenAiListwiseLlmRanker(LlmRanker):
    def __init__(self, model_name_or_path, api_key, window_size, step_size, num_repeat):
        self.llm = model_name_or_path
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.window_size = window_size
        self.step_size = step_size
        self.num_repeat = num_repeat
        openai.api_key = api_key

    def compare(self, query: str, docs: List):
        messages = create_permutation_instruction_chat(query, docs, self.llm)
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=self.llm,
                    messages=messages,
                    temperature=0.0,
                    request_timeout=15)

                return completion['choices'][0]['message']['content']
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:

        for _ in range(self.num_repeat):
            ranking = copy.deepcopy(ranking)
            end_pos = len(ranking)
            start_pos = end_pos - self.window_size
            while start_pos >= 0:
                start_pos = max(start_pos, 0)
                result = self.compare(query, ranking[start_pos: end_pos])
                ranking = receive_permutation(ranking, result, start_pos, end_pos)
                end_pos = end_pos - self.step_size
                start_pos = start_pos - self.step_size

        for i, doc in enumerate(ranking):
            doc.score = -i
        return ranking

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])


def prompt_generator_t5(prompt_type, original_prompt_number, instruction, output, tone, order, position, role, query,
                        docs):
    num = len(docs)
    rank = 0
    passages = '\n'
    for doc in docs:
        rank += 1
        content = doc.text
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        content = ' '.join(content.split()[:300])
        passages += f"[{rank}] {content}\n\n"

    if prompt_type == "original":
        original_prompt_dict = {
            1: passages + f"Query = {query}\n\n" +
               "Passages = [Passage 1, Passage2, Passage3, Passage4] \n"
               "Sort the Passages by their relevance to the Query. \n"
               "Sorted Passages = [",
            2: f"You are RankGPT, "
               f"an intelligent assistant that can rank passages based on their relevancy to the query. "
               f"I will provide you with {num} passages, each indicated by number identifier []. \n"
               f"Rank the passages based on their relevance to query: {query}\n" + passages +
               f"Search Query: {query}. \n"
               f"Rank the {num} passages above based on their relevance to the search query. "
               f"The passages should be listed in descending order using identifiers. "
               f"The most relevant passages should be listed first. "
               f"The output format should be [] > [], e.g., [1] > [2]. "
               f"Only response the ranking results, do not say any word or explain."
        }
        return original_prompt_dict[original_prompt_number]

    elif prompt_type == "adjusted":
        instruction_dict = {
            "instruction_1": f"Search Query: {query}. \n"
                             f"Rank the {num} passages above based on their relevance to the search query.\n",
            "instruction_2": f"I will provide you with {num} passages, each indicated by number identifier []. \n"
                             f"Rank the passages based on their relevance to query: {query}\n",
            "instruction_3": f"Sort the Passages by their relevance to the Query. \nQuery = {query}\n"
        }

        output_dict = {
            "output_1": f"The passages should be listed in descending order using identifiers. "
                        f"The most relevant passages should be listed first. "
                        f"The output format should be [] > [], e.g., [1] > [2]. ",
            "output_2": "Sorted Passages = [ "
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
        passages = passages

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
                             docs):
    num = len(docs)
    rank = 0
    passages = '\n'
    for doc in docs:
        rank += 1
        content = doc.text
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        content = ' '.join(content.split()[:300])
        passages += f"[{rank}] {content}\n\n"

    if prompt_type == "original":
        original_prompt_dict = {
            1: passages + f"Query = {query}\n\n" +
               "Passages = [Passage 1, Passage2, Passage3, Passage4] \n"
               "Sort the Passages by their relevance to the Query. \n"
               "Sorted Passages = [",
            2: f"You are RankGPT, "
               f"an intelligent assistant that can rank passages based on their relevancy to the query. "
               f"I will provide you with {num} passages, each indicated by number identifier []. \n"
               f"Rank the passages based on their relevance to query: {query}\n" + passages +
               f"Search Query: {query}. \n"
               f"Rank the {num} passages above based on their relevance to the search query. "
               f"The passages should be listed in descending order using identifiers. "
               f"The most relevant passages should be listed first. "
               f"The output format should be [] > [], e.g., [1] > [2]. "
               f"Only response the ranking results, do not say any word or explain."
        }
        return original_prompt_dict[original_prompt_number]

    elif prompt_type == "adjusted":
        instruction_dict = {
            "instruction_1": f"Search Query: {query}. \n"
                             f"Rank the {num} passages above based on their relevance to the search query.\n",
            "instruction_2": f"I will provide you with {num} passages, each indicated by number identifier []. \n"
                             f"Rank the passages based on their relevance to query: {query}\n",
            "instruction_3": f"Sort the Passages by their relevance to the Query. \nQuery = {query}\n"
        }

        output_dict = {
            "output_1": f"The passages should be listed in descending order using identifiers. "
                        f"The most relevant passages should be listed first. "
                        f"The output format should be [] > [], e.g., [1] > [2]. ",
            "output_2": "Sorted Passages = [ "
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
        passages = passages

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


class ListwiseLlmRanker(OpenAiListwiseLlmRanker):
    CHARACTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                  "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
                  "W"]  # "Passage X" and "Passage Y" will be tokenized into 3 tokens, so we dont use for now

    def __init__(self,
                 model_name_or_path,
                 tokenizer_name_or_path,
                 device, window_size,
                 step_size,
                 scoring='generation',
                 num_repeat=1,
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

        self.scoring = scoring
        self.device = device
        self.window_size = window_size
        self.step_size = step_size
        self.num_repeat = num_repeat

        self.HF_TOKEN = hf_token

        # print(AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir, token=self.HF_TOKEN).model_type)
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir, token=self.HF_TOKEN)

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
                                                           add_special_tokens=False).to(
                self.device) if self.tokenizer else None
            self.target_token_ids = self.tokenizer.batch_encode_plus([f'<pad> Passage {self.CHARACTERS[i]}'
                                                                      for i in range(len(self.CHARACTERS))],
                                                                     return_tensors="pt",
                                                                     add_special_tokens=False,
                                                                     padding=True).input_ids[:, -1]

        elif self.config.model_type == 'llama':
            # self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            # self.tokenizer.use_default_system_prompt = False
            # if 'vicuna' and 'v1.5' in model_name_or_path:
            #     self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
            #
            # self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
            #                                                 device_map='auto',
            #                                                 torch_dtype=torch.float16 if device == 'cuda'
            #                                                 else torch.float32,
            #                                                 cache_dir=cache_dir).eval()

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
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
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

        else:
            raise NotImplementedError

    def compare(self, query: str, docs: List):
        if self.scoring == 'generation':

            if self.config.model_type == 't5':
                input_text = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
                                                 self.output, self.tone, self.order, self.position, self.role, query,
                                                 docs)

                if not self.has_printed_input_text:
                    # print("prompt is \n" + input_text + '\n')
                    # Set the flag to True after printing, to prevent future prints
                    self.has_printed_input_text = True

                input_ids = self.tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.to(self.device)

                output_ids = self.llm.generate(input_ids)[0]
                output = self.tokenizer.decode(output_ids,
                                               skip_special_tokens=True).strip()
                # print(output)

            elif self.config.model_type == 'llama' or self.config.model_type == 'mistral':

                input_text = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
                                                 self.output, self.tone, self.order, self.position, self.role, query,
                                                 docs)

                conversation = [{"role": "user", "content": input_text}]

                prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                # print("prompt is \n" + prompt + '\n')

                input_ids = self.tokenizer(prompt, return_tensors="pt", padding='longest').input_ids.to(self.device)

                output_ids = self.llm.generate(input_ids, max_new_tokens=100, do_sample=True)[0]
                output = self.tokenizer.decode(output_ids[input_ids.shape[1]:],
                                               skip_special_tokens=True).strip()
                # store output to output_text_mistal.txt
                with open(f"output_text_{self.config.model_type}.txt", "a") as f:
                    f.write(output + '\n' + "Next\n")
                # statistics for length of output
                with open(f"output_length_{self.config.model_type}.txt", "a") as f:
                    f.write(str(len(output)) + '\n')

            elif self.config.model_type == 'gemma':

                input_text = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
                                                 self.output, self.tone, self.order, self.position, self.role, query,
                                                 docs)

                conversation = [{"role": "user", "content": input_text}]

                prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                # print("prompt is \n" + prompt + '\n')

                input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(
                    self.device)

                output_ids = self.llm.generate(input_ids, max_new_tokens=100, do_sample=True)[0]
                output = self.tokenizer.decode(output_ids[input_ids.shape[1]:],
                                               skip_special_tokens=True).strip()

            # elif self.config.model_type == 'mistral':
            #
            #     prompt = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
            #                                       self.output, self.tone, self.order, self.position, self.role, query,
            #                                       docs)
            #
            #     conversation = [{"role": "user", "content": prompt}]
            #
            #     input_text = self.tokenizer.apply_chat_template(conversation, tokenize=False,
            #                                                     add_generation_prompt=True)
            #
            #     input_text += "Passage label:"
            #
            #     print("\nPrompt is\n" + input_text)
            #
            #     input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            #
            #     output_ids = self.llm.generate(input_ids,
            #                                    max_new_tokens=20,
            #                                    do_sample=False,
            #                                    temperature=0.0,
            #                                    top_p=None)[0]
            #
            #     output = self.tokenizer.decode(output_ids[input_ids.shape[1]:],
            #                                    skip_special_tokens=True).strip()
            #     print(self.config.model_type, "\noutput is \n" + output + " \n")


        elif self.scoring == 'likelihood':
            passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])
            input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                         + passages + '\n\nOutput only the passage label of the most relevant passage:'

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

            with torch.no_grad():
                logits = self.llm(input_ids=input_ids, decoder_input_ids=self.decoder_input_ids).logits[0][-1]
                distributions = torch.softmax(logits, dim=0)
                scores = distributions[self.target_token_ids[:len(docs)]]
                ranked = sorted(zip([f"[{str(i + 1)}]" for i in range(len(docs))], scores), key=lambda x: x[1],
                                reverse=True)
                output = '>'.join(ranked[i][0] for i in range(len(ranked)))

        return output

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])
