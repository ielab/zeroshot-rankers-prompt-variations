import os
import csv
import logging
import ir_datasets
# from pyserini.search.lucene import LuceneSearcher
# from pyserini.search._base import get_topics
import run
from rankers.rankers import SearchResult
from rankers.pointwise import PointwiseLlmRanker
from rankers.setwise import SetwiseLlmRanker, OpenAiSetwiseLlmRanker
from rankers.pairwise import PairwiseLlmRanker, OpenAiPairwiseLlmRanker
from rankers.listwise import OpenAiListwiseLlmRanker, ListwiseLlmRanker
from tqdm import tqdm
import argparse
import sys
import json
import time
import random

random.seed(929)
logger = logging.getLogger(__name__)


def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


def write_run_file(path, results, tag):
    with open(path, 'w') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{tag}\n")
                rank += 1


def main(args):

    if args.pointwise:
        ranker = PointwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                    tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                    device=args.run.device,
                                    cache_dir=args.run.cache_dir,
                                    method=args.pointwise.method,
                                    batch_size=args.pointwise.batch_size,
                                    hf_token=args.run.hf_token,
                                    prompt_type=args.pointwise.prompt_type,
                                    original_prompt_number=args.pointwise.original_prompt_number,
                                    instruction=args.pointwise.instruction,
                                    output=args.pointwise.output,
                                    tone=args.pointwise.tone,
                                    order=args.pointwise.order,
                                    position=args.pointwise.position,
                                    role=args.pointwise.role)

    elif args.setwise:
        if args.run.openai_key:
            ranker = OpenAiSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                            api_key=args.run.openai_key,
                                            num_child=args.setwise.num_child,
                                            method=args.setwise.method,
                                            k=args.setwise.k)
        else:
            ranker = SetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                      tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                      device=args.run.device,
                                      cache_dir=args.run.cache_dir,
                                      num_child=args.setwise.num_child,
                                      scoring=args.run.scoring,
                                      method=args.setwise.method,
                                      num_permutation=args.setwise.num_permutation,
                                      k=args.setwise.k,
                                      hf_token=args.run.hf_token,
                                      prompt_type = args.setwise.prompt_type,
                                      original_prompt_number = args.setwise.original_prompt_number,
                                      instruction = args.setwise.instruction,
                                      output = args.setwise.output,
                                      tone = args.setwise.tone,
                                      order = args.setwise.order,
                                      position = args.setwise.position,
                                      role = args.setwise.role)

    elif args.pairwise:
        if args.pairwise.method != 'allpair':
            args.pairwise.batch_size = 2
            logger.info(f'Setting batch_size to 2.')

        if args.run.openai_key:
            ranker = OpenAiPairwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             api_key=args.run.openai_key,
                                             method=args.pairwise.method,
                                             k=args.pairwise.k)
        else:
            ranker = PairwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                       tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                       device=args.run.device,
                                       cache_dir=args.run.cache_dir,
                                       method=args.pairwise.method,
                                       batch_size=args.pairwise.batch_size,
                                       k=args.pairwise.k,
                                       hf_token=args.run.hf_token,
                                       prompt_type=args.pairwise.prompt_type,
                                       original_prompt_number=args.pairwise.original_prompt_number,
                                       instruction=args.pairwise.instruction,
                                       output=args.pairwise.output,
                                       tone=args.pairwise.tone,
                                       order=args.pairwise.order,
                                       position=args.pairwise.position,
                                       role=args.pairwise.role)

    elif args.listwise:
        if args.run.openai_key:
            ranker = OpenAiListwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             api_key=args.run.openai_key,
                                             window_size=args.listwise.window_size,
                                             step_size=args.listwise.step_size,
                                             num_repeat=args.listwise.num_repeat)
        else:
            ranker = ListwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                       tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                       device=args.run.device,
                                       cache_dir=args.run.cache_dir,
                                       window_size=args.listwise.window_size,
                                       step_size=args.listwise.step_size,
                                       scoring=args.run.scoring,
                                       num_repeat=args.listwise.num_repeat,
                                       hf_token=args.run.hf_token,
                                       prompt_type=args.listwise.prompt_type,
                                       original_prompt_number=args.listwise.original_prompt_number,
                                       instruction=args.listwise.instruction,
                                       output=args.listwise.output,
                                       tone=args.listwise.tone,
                                       order=args.listwise.order,
                                       position=args.listwise.position,
                                       role=args.listwise.role)
    else:
        raise ValueError('Must specify either --pointwise, --setwise, --pairwise or --listwise.')

    query_map = {}

    if args.run.ir_dataset_name is not None:
        dataset = ir_datasets.load(args.run.ir_dataset_name)
        for query in dataset.queries_iter():
            qid = query.query_id
            text = query.text
            query_map[qid] = ranker.truncate(text, args.run.query_length)
        dataset = ir_datasets.load(args.run.ir_dataset_name)
        docstore = dataset.docs_store()
    else:
        topics = get_topics(args.run.pyserini_index+'-test')
        for topic_id in list(topics.keys()):
            text = topics[topic_id]['title']
            query_map[str(topic_id)] = ranker.truncate(text, args.run.query_length)
        docstore = LuceneSearcher.from_prebuilt_index(args.run.pyserini_index+'.flat')

    logger.info(f'Loading first stage run from {args.run.run_path}.')
    first_stage_rankings = []
    with open(args.run.run_path, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f):
            qid, _, docid, _, score, _ = line.strip().split()

            if args.run.ir_dataset_name == "msmarco-passage/trec-dl-2020":
                # Get the current working directory
                current_path = os.path.dirname(os.path.abspath(__file__))
                # Construct the full path to the queries.dl2020.tsv file
                file_path = os.path.join(current_path, "queries.dl2020.tsv")
                # Read the file
                with open(file_path, 'r') as f:
                    reader = csv.reader(f, delimiter='\t')
                    first_column = [row[0] for row in reader]
                if qid != current_qid:
                    if current_qid is not None and current_qid in first_column:
                        first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.run.hits]))
                    current_ranking = []
                    current_qid = qid
                if len(current_ranking) >= args.run.hits:
                    continue
                if args.run.ir_dataset_name is not None:
                    text = docstore.get(docid).text
                    if 'title' in dir(docstore.get(docid)):
                        text = f'{docstore.get(docid).title} {text}'
                else:
                    data = json.loads(docstore.doc(docid).raw())
                    text = data['text']
                    if 'title' in data:
                        text = f'{data["title"]} {text}'
                text = ranker.truncate(text, args.run.passage_length)
                current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))

            else:
                if qid != current_qid:
                    if current_qid is not None:
                        first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.run.hits]))
                    current_ranking = []
                    current_qid = qid
                if len(current_ranking) >= args.run.hits:
                    continue
                if args.run.ir_dataset_name is not None:
                    text = docstore.get(docid).text
                    if 'title' in dir(docstore.get(docid)):
                        text = f'{docstore.get(docid).title} {text}'
                else:
                    data = json.loads(docstore.doc(docid).raw())
                    text = data['text']
                    if 'title' in data:
                        text = f'{data["title"]} {text}'
                text = ranker.truncate(text, args.run.passage_length)
                current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))

        first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.run.hits]))

    reranked_results = []

    for qid, query, ranking in tqdm(first_stage_rankings):
        if args.run.shuffle_ranking is not None:
            if args.run.shuffle_ranking == 'random':
                random.shuffle(ranking)
            elif args.run.shuffle_ranking == 'inverse':
                ranking = ranking[::-1]
            else:
                raise ValueError(f'Invalid shuffle ranking method: {args.run.shuffle_ranking}.')
        reranked_results.append((qid, query, ranker.rerank(query, ranking)))

    write_run_file(args.run.save_path, reranked_results, 'LLMRankers')
    print(f'Reranked run saved to {args.run.save_path} \n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='sub-commands')

    run_parser = commands.add_parser('run')
    run_parser.add_argument('--run_path', type=str, help='Path to the first stage run file (TREC format) to rerank.')
    run_parser.add_argument('--save_path', type=str, help='Path to save the reranked run file (TREC format).')
    run_parser.add_argument('--model_name_or_path', type=str,
                            help='Path to the pretrained model or model identifier from huggingface.co/models')
    run_parser.add_argument('--tokenizer_name_or_path', type=str, default=None,
                            help='Path to the pretrained tokenizer or tokenizer identifier from huggingface.co/tokenizers')
    run_parser.add_argument('--ir_dataset_name', type=str, default=None)
    run_parser.add_argument('--pyserini_index', type=str, default=None)
    run_parser.add_argument('--hits', type=int, default=100)
    run_parser.add_argument('--query_length', type=int, default=128)
    run_parser.add_argument('--passage_length', type=int, default=128)
    run_parser.add_argument('--device', type=str, default='cuda')
    run_parser.add_argument('--cache_dir', type=str, default=None)
    run_parser.add_argument('--openai_key', type=str, default=None)
    run_parser.add_argument('--scoring', type=str, default='generation', choices=['generation', 'likelihood'])
    run_parser.add_argument('--shuffle_ranking', type=str, default=None, choices=['inverse', 'random'])
    run_parser.add_argument('--hf_token', type=str, default=None)

    pointwise_parser = commands.add_parser('pointwise')
    pointwise_parser.add_argument('--method', type=str, default='yes_no',
                                  choices=['qlm', 'yes_no'])
    pointwise_parser.add_argument('--batch_size', type=int, default=2)
    pointwise_parser.add_argument('--prompt_type', type=str, default="adjusted", choices=['original', 'adjusted'])
    pointwise_parser.add_argument('--original_prompt_number', type=int, default=1)
    pointwise_parser.add_argument('--instruction', type=str, default="instruction_1")
    pointwise_parser.add_argument('--output', type=str, default="output_1")
    pointwise_parser.add_argument('--tone', type=str, default="tone_1")
    pointwise_parser.add_argument('--order', type=str, default="query_first", choices=['query_first', 'passage_first'])
    pointwise_parser.add_argument('--position', type=str, default="beginning", choices=['beginning', 'ending'])
    pointwise_parser.add_argument('--role', type=str, default="False", choices=['True', 'False'])

    pairwise_parser = commands.add_parser('pairwise')
    pairwise_parser.add_argument('--method', type=str, default='allpair',
                                 choices=['allpair', 'heapsort', 'bubblesort'])
    pairwise_parser.add_argument('--batch_size', type=int, default=2)
    pairwise_parser.add_argument('--k', type=int, default=10)
    pairwise_parser.add_argument('--prompt_type', type=str, default="adjusted", choices=['original', 'adjusted'])
    pairwise_parser.add_argument('--original_prompt_number', type=int, default=1)
    pairwise_parser.add_argument('--instruction', type=str, default="instruction_1")
    pairwise_parser.add_argument('--output', type=str, default="output_1")
    pairwise_parser.add_argument('--tone', type=str, default="tone_1")
    pairwise_parser.add_argument('--order', type=str, default="query_first", choices=['query_first', 'passage_first'])
    pairwise_parser.add_argument('--position', type=str, default="beginning", choices=['beginning', 'ending'])
    pairwise_parser.add_argument('--role', type=str, default="False", choices=['True', 'False'])

    setwise_parser = commands.add_parser('setwise')
    setwise_parser.add_argument('--num_child', type=int, default=3)
    setwise_parser.add_argument('--method', type=str, default='heapsort',
                                choices=['heapsort', 'bubblesort'])
    setwise_parser.add_argument('--k', type=int, default=10)
    setwise_parser.add_argument('--num_permutation', type=int, default=1)
    setwise_parser.add_argument('--prompt_type', type=str, default="adjusted", choices=['original', 'adjusted'])
    setwise_parser.add_argument('--original_prompt_number', type=int, default=1)
    setwise_parser.add_argument('--instruction', type=str, default="instruction_1")
    setwise_parser.add_argument('--output', type=str, default="output_1")
    setwise_parser.add_argument('--tone', type=str, default="tone_1")
    setwise_parser.add_argument('--order', type=str, default="query_first",choices=['query_first', 'passage_first'])
    setwise_parser.add_argument('--position', type=str, default="beginning", choices=['beginning', 'ending'])
    setwise_parser.add_argument('--role', type=str, default="False", choices=['True', 'False'])

    listwise_parser = commands.add_parser('listwise')
    listwise_parser.add_argument('--window_size', type=int, default=3)
    listwise_parser.add_argument('--step_size', type=int, default=1)
    listwise_parser.add_argument('--num_repeat', type=int, default=1)
    listwise_parser.add_argument('--prompt_type', type=str, default="adjusted", choices=['original', 'adjusted'])
    listwise_parser.add_argument('--original_prompt_number', type=int, default=1)
    listwise_parser.add_argument('--instruction', type=str, default="instruction_1")
    listwise_parser.add_argument('--output', type=str, default="output_1")
    listwise_parser.add_argument('--tone', type=str, default="tone_1")
    listwise_parser.add_argument('--order', type=str, default="query_first", choices=['query_first', 'passage_first'])
    listwise_parser.add_argument('--position', type=str, default="beginning", choices=['beginning', 'ending'])
    listwise_parser.add_argument('--role', type=str, default="False", choices=['True', 'False'])

    args = parse_args(parser, commands)

    if args.run.ir_dataset_name is not None and args.run.pyserini_index is not None:
        raise ValueError('Must specify either --ir_dataset_name or --pyserini_index, not both.')

    arg_dict = vars(args)
    if arg_dict['run'] is None or sum(arg_dict[arg] is not None for arg in arg_dict) != 2:
        raise ValueError('Need to set --run and can only set one of --pointwise, --pairwise, --setwise, --listwise')
    main(args)
