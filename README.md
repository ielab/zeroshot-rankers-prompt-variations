# zeroshot-rankers-prompt-variations

## Citation
If you find this code useful for your research, please consider citing our ECIR 2025 paper: "An Investigation of Prompt Variations for Zero-shot LLM-based Rankers".

BibTeX entry:
```
@inproceedings{sun2025investigation,
  title = {An Investigation of Prompt Variations for Zero-shot LLM-based Rankers},
  author = {Sun, Shuoqi and Zhuang, Shengyao and Wang, Shuai and Zuccon, Guido},
  editor = {Hauff, Claudia and Macdonald, Craig and Jannach, Dietmar and Kazai, Gabriella and Nardini, Franco Maria and Pinelli, Fabio and Silvestri, Fabrizio and Tonellotto, Nicola},
  booktitle = {Advances in Information Retrieval},
  year = {2025},
  publisher = {Springer Nature Switzerland},
  address = {Cham},
  pages = {185--201},
  isbn = {978-3-031-88711-6},
  doi = {10.1007/978-3-031-88711-6_12},
  url = {https://doi.org/10.1007/978-3-031-88711-6_12},
}
```

## Large Language Models Used in This Research
| LLMs             | Name in Hugging Face              |
|------------------|-----------------------------------|
| FlanT5-Large     | google/flan-t5-large              |
| FlanT5-XL        | google/flan-t5-xl                 |
| FlanT5-XXL       | google/flan-t5-xxl                |
| Mistral-7B-v0.2  | mistralai/Mistral-7B-Instruct-v0.2|
| LLama3-8B        | meta-llama/Meta-Llama-3-8B-Instruct|

*Note: Some models require Hugging Face tokens for access. Ensure you have the appropriate authorization.*

## Datasets Used in This Research
| Datasets                        | Name in ir_dataset Library       |
|---------------------------------|----------------------------------|
| TREC Deep Learning (DL) 2019    | msmarco-passage/trec-dl-2019     |
| TREC Deep Learning (DL) 2020    | msmarco-passage/trec-dl-2020     |
| BeIR TREC COVID                 | beir/trec-covid                  |

## How to Run `Run.py`
Here is a step-by-step guide to run the Listwise script as an example:

### 1. Clone the Repository
#### 2.Set Up the Environment
Create and activate a Conda environment using the provided "environment.yml" file. You can create the environment with the following commands:
```
cd path_of_the_code
conda env create -n my_env -f environment.yml
conda activate my_env
```
or if you want to create it in a specific path:
```
cd path_of_the_code
conda create --prefix path_for_conda/envs -f environment.yml
conda activate path_for_conda/envs
```
#### 3.Copy following script to your command line after you properly modify the variables defined.
Copy the following script to your command line after modifying the variables as needed.

Variable Explanations:
- **server_path**: The directory where the project code is stored.
- **model**: The large language model to use. Choose from the LLMs table.
- **ir_dataset_name**: The dataset to use, chosen from the Datasets table.
- **hf_token**: Your huggingface token to access models.
- **other configuration**: Settings for the prompt. Options for listwise include:
```
instructions=("instruction_1" "instruction_2" "instruction_3")
outputs=("output_1" "output_2")
tones=("tone_0" "tone_1" "tone_2" "tone_3" "tone_4" "tone_5")
orders=("query_first" "passage_first")
positions=("beginning" "ending")
roles=("True" "False")
```
After variables defination, you can run this script. Evetually, you will get your running result in "run" folder and cache in "cache" folder in your "server_path".
```
#!/bin/bash
# activate your environment here or before running this script
server_path= "YOUR_ROOT_PATH_HERE/zeroshot-rankers-prompt-variations" # change this as what you have in your server
model="mistralai/Mistral-7B-Instruct-v0.2" # modify your model here
ir_dataset_name="msmarco-passage/trec-dl-2019" # modify your dataset here

instruction="instruction_1"
output="output_1"
order="query_first"
position="beginning"
tone="tone_0"
role="True"

$hf_token=Your_HF_Token

cd $server_path
python_script_path="$server_path/run.py" # Path to the python script (run.py), no need to change, just change server_path
export IR_DATASETS_HOME="$server_path/ir_datasets"
# shellcheck disable=SC2164
cache_dir="$server_path/cache"

# control run essentials
query_length=20
passage_length=80

reranker="listwise"
output_path="$server_path/runs/$model/$ir_dataset_name/$reranker"
mkdir -p "$output_path"
output_file_path="$output_path/${instruction}_${output}_${tone}_${order}_${position}_${role}.txt"

python3 "$python_script_path" \
run --model_name_or_path $model \
--tokenizer_name_or_path $model \
--run_path "$run_path" \
--save_path "$output_file_path" \
--ir_dataset_name $ir_dataset_name \
--hits 100 \
--query_length $query_length \
--passage_length $passage_length \
--scoring "generation" \
--device "cuda" \
--cache_dir $cache_dir \
--hf_token $hf_token
$reranker --window_size 4 \
--step_size 2 \
--num_repeat 5 \
--instruction "$instruction" \
--output "$output" \
--tone "$tone" \
--order "$order" \
--position "$position" \
--role "$role"
```

## How to Run All Rankers?

***The script varies slightly for different rankers due to their unique features, but they share common commands at the start. Include these commands at the top of your script:***
```
#!/bin/bash
# activate your environment here or before running this script
server_path= "YOUR_ROOT_PATH_HERE/prompt-meta-code" # change this as what you have in your server
model="mistralai/Mistral-7B-Instruct-v0.2" # modify your model here
ir_dataset_name="msmarco-passage/trec-dl-2019" # modify your dataset here

instruction="instruction_1"
output="output_1"
order="query_first"
position="beginning"
tone="tone_0"
role="True"

$hf_token=Your_HF_Token

cd $server_path
python_script_path="$server_path/run.py" # Path to the python script (run.py), no need to change, just change server_path
export IR_DATASETS_HOME="$server_path/ir_datasets"
# shellcheck disable=SC2164
cache_dir="$server_path/cache"

# control run essentials
query_length=20
passage_length=80
```
### Pointwise
Setting options for a Pointwise ranker:
```
instructions=("instruction_1" "instruction_2" "instruction_3" "instruction_4")
outputs=("output_1" "output_2" "output_3" "output_4")
tones=("tone_0" "tone_1" "tone_2" "tone_3" "tone_4" "tone_5")
orders=("query_first" "passage_first")
positions=("beginning" "ending")
roles=("True" "False")
```
Put these command after the common part:
```
reranker="pointwise"
output_path="$server_path/runs/$model/$ir_dataset_name/$reranker"
mkdir -p "$output_path"
output_file_path="$output_path/${instruction}_${output}_${tone}_${order}_${position}_${role}.txt"
python3 "$python_script_path" \
run --model_name_or_path $model \
--tokenizer_name_or_path $model \
--run_path "$run_path" \
--save_path "$output_file_path" \
--ir_dataset_name $ir_dataset_name \
--hits 100 \
--query_length $query_length \
--passage_length $passage_length \
--scoring "generation" \
--device "cuda" \
--cache_dir $cache_dir \
$reranker --method yes_no \
--batch_size 32 \
--instruction "$instruction" \
--output "$output" \
--tone "$tone" \
--order "$order" \
--position "$position" \
--role "$role"
```
*Please notice if you have limited memory, you may adjust batch_size to smaller numbers such as 16 or 8 to fit in your cuda device.*

### Pairwise
Setting options for a Pairwise ranker:
```
instructions=("instruction_1")
outputs=("output_1")
tones=("tone_0" "tone_1" "tone_2" "tone_3" "tone_4" "tone_5")
orders=("query_first" "passage_first")
positions=("beginning" "ending")
roles=("True" "False")
```
Put these command after the common part:
```
reranker="pairwise"
output_path="$server_path/runs/$model/$ir_dataset_name/$reranker"
mkdir -p "$output_path"
output_file_path="$output_path/${instruction}_${output}_${tone}_${order}_${position}_${role}.txt"
python3 "$python_script_path" \
run --model_name_or_path $model \
--tokenizer_name_or_path $model \
--run_path "$run_path" \
--save_path "$output_file_path" \
--ir_dataset_name $ir_dataset_name \
--hits 100 \
--query_length $query_length \
--passage_length $passage_length \
--scoring "generation" \
--device "cuda" \
--cache_dir $cache_dir \
$reranker --method heapsort \
--k 10 \
--instruction "$instruction" \
--output "$output" \
--tone "$tone" \
--order "$order" \
--position "$position" \
--role "$role"
```

### Listwise
Setting options for a Listwise ranker:
```
instructions=("instruction_1" "instruction_2" "instruction_3" "instruction_4")
outputs=("output_1" "output_2" "output_3" "output_4")
tones=("tone_0" "tone_1" "tone_2" "tone_3" "tone_4" "tone_5")
orders=("query_first" "passage_first")
positions=("beginning" "ending")
roles=("True" "False")
```
Put these command after the common part:
```
reranker="listwise"
output_path="$server_path/runs/$model/$ir_dataset_name/$reranker"
mkdir -p "$output_path"
output_file_path="$output_path/${instruction}_${output}_${tone}_${order}_${position}_${role}.txt"
python3 "$python_script_path" \
run --model_name_or_path $model \
--tokenizer_name_or_path $model \
--run_path "$run_path" \
--save_path "$output_file_path" \
--ir_dataset_name $ir_dataset_name \
--hits 100 \
--query_length $query_length \
--passage_length $passage_length \
--scoring "generation" \
--device "cuda" \
--cache_dir $cache_dir \
$reranker --window_size 4 \
--step_size 2 \
--num_repeat 5 \
--instruction "$instruction" \
--output "$output" \
--tone "$tone" \
--order "$order" \
--position "$position" \
--role "$role"
```

### Setwise
Setting options for a Setwise ranker:
```
instructions=("instruction_1")
outputs=("output_1" "output_2" "output_3")
tones=("tone_0" "tone_1" "tone_2" "tone_3" "tone_4" "tone_5")
orders=("query_first" "passage_first")
positions=("beginning" "ending")
roles=("True" "False")
```
Put these command after the common part:
```
reranker="setwise"
output_path="$server_path/runs/$model/$ir_dataset_name/$reranker"
mkdir -p "$output_path"
output_file_path="$output_path/${instruction}_${output}_${tone}_${order}_${position}_${role}.txt"
python3 "$python_script_path" \
run --model_name_or_path $model \
--tokenizer_name_or_path $model \
--run_path "$run_path" \
--save_path "$output_file_path" \
--ir_dataset_name $ir_dataset_name \
--hits 100 \
--query_length $query_length \
--passage_length $passage_length \
--scoring "generation" \
--device "cuda" \
--cache_dir $cache_dir \
$reranker --num_child 2 \
--method "heapsort" \
--k 10 \
--instruction "$instruction" \
--output "$output" \
--tone "$tone" \
--order "$order" \
--position "$position" \
--role "$role"
```

## How to Evaluate Running results?
You can evaluate running results by using "pyserini.eval.trec_eval" library. Here is an example command:
```
python -m pyserini.eval.trec_eval -m ndcg_cut.10 dl19-passage "Your_Run_File_Path" 
```
If you want to store the eveluation results into a file:
```
python -m pyserini.eval.trec_eval -m ndcg_cut.10 dl19-passage "Your_Run_File_Path" > $Evaluation_File_Path
```
Or if you also care about the running results for each single query:
```
python -m pyserini.eval.trec_eval -q -m ndcg_cut.10 dl19-passage "Your_Run_File_Path" > $Evaluation_File_Path
```

*Please notice you should choose the corresponding dataset to evluate your run files. For example, if you use "msmarco-passage/trec-dl-2019" as your ir_dataset name for runnings, you should use "dl19-passage" in your command to evluate them.*

| Datasets | Name of Qrels Files |
|-----------------|-----------------|
|TREC Deep Learning (DL) 2019   | dl19-passage|
|TREC Deep Learning (DL) 2020  |dl20-passage |
|BeIR TREC COVID  | beir-v1.0.0-trec-covid-test|
||
