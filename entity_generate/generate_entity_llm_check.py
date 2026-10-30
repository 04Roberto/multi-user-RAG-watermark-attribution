from html import parser
import json
import os
import time
import pickle
import traceback
from datasets import load_dataset

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import sys
import argparse
import random
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print(project_root)
sys.path.append(project_root)
from src.utils import load_beir_datasets, load_json, save_json

from src.utils import Log, file_exist
def data_prepare(dataname, dataset):
    ndataset = {}
    if dataname in ['nq', 'msmarco', 'hotpotqa', 'nfcorpus', 'trec-covid']:
        for key, value in dataset.items():
            ndataset[key] = value['text']
    elif dataname == 'closed_qa':
        for i, item in enumerate(dataset):
            ndataset[str(i)] = f"{item['instruction']}. {item['context']}"
    return ndataset


def generate_entity(dataset, basepath, prob=0.01):
    entities_dict = {}
    node_type_dict = {}
    relation_type_dict = {}
    unique_relation = []
    current_key = ''
    tcount =0
    ## load has saved results
    try:
        entities_dict, node_type_dict, relation_type_dict, unique_relation,  current_key = load_checkpoint(basepath)
        print(current_key,type(current_key))
        print("successfully load saved file")
    except Exception as e:
        print(f"Error occurred while converting document to graph: {e}")
        traceback.print_exc()
             
    # 按 key 排序遍历
    for i, key in enumerate(sorted(dataset.keys())):
        if current_key != '' and key < current_key:
            continue

        if random.random() > prob:
            continue    
        
        item = dataset[key]
        print(f"Processing item {i}: key = {key}, tcount: {tcount}")
        tcount += 1
        
        # We wrap the transformation in a try-except block to catch malformed LLM outputs
        try:
            documents = [Document(page_content=item)]
            # print(f'documents:{documents}')
            
            # This is where the "Unhashable type" and "ValidationError" happen
            graph_documents = llm_transformer.convert_to_graph_documents(documents)
            
            if graph_documents and len(graph_documents) > 0:
                nodes_list = graph_documents[0].nodes
                # Continue with your logic for nodes_list here...
            else:
                print(f"Warning: No graph documents generated for key {key}")
                continue

        except Exception as e:
            # This catches the TypeError (unhashable) and ValidationError (Pydantic)
            print(f"Error occurred while converting document {key} to graph: {e}")
            # Optional: print the full error for debugging if needed
            # import traceback
            # traceback.print_exc()
            continue # This is the most important line: it keeps the loop running!

        # 处理 nodes
        # print(f'node_list:{nodes_list}')
        for node in nodes_list:
            node_type_dict[node.type] = node_type_dict.get(node.type, 0) + 1
            if node.id not in entities_dict:
                entities_dict[node.id] = node.type

        relations_list = graph_documents[0].relationships

        # 处理 relations
        for relation in relations_list:
            relation_type_dict[relation.type] = relation_type_dict.get(relation.type, 0) + 1
            if [relation.source.id, relation.target.id, relation.type] not in unique_relation:
                unique_relation.append([relation.source.id, relation.target.id, relation.type])

        if tcount % 20  == 1 :
            print(i, time.time())
            
            # 每次处理完当前 key，存储当前中间结果
            save_checkpoint(entities_dict, node_type_dict, relation_type_dict, unique_relation , key, basepath)
            # break
    save_checkpoint(entities_dict, node_type_dict, relation_type_dict, unique_relation,   key, basepath)
    print(f'all dataset has been processed')
 

    return True


def save_checkpoint(entities_dict, node_type_dict, relation_type_dict, unique_relation,  current_key, basepath):


    # Define paths for saving the outputs
    entities_dict_path = os.path.join(basepath,   'entities_dict_llm.json')
    entity_type_path = os.path.join(basepath,   'entity_type_llm.json')
    relation_type_path = os.path.join(basepath,  'relation_type_llm.json')
    relation_list_path = os.path.join(basepath,  'relation_list_llm.json')

    checkpoint_file = os.path.join(basepath,  'checkpoint.json')
    save_json( current_key, checkpoint_file)

    save_json(entities_dict, entities_dict_path)
    save_json(node_type_dict, entity_type_path)
    save_json(relation_type_dict, relation_type_path)
    save_json( unique_relation, relation_list_path)

def load_checkpoint(  basepath):
 
    # Define paths for saving the outputs
    entities_dict_path = os.path.join(basepath,   'entities_dict_llm.json')
    entity_type_path = os.path.join(basepath,   'entity_type_llm.json')
    relation_type_path = os.path.join(basepath,  'relation_type_llm.json')
    relation_list_path = os.path.join(basepath,  'relation_list_llm.json')
    checkpoint_file = os.path.join(basepath,  'checkpoint.json')


    entities_dict = load_json(entities_dict_path)
    node_type_dict = load_json(entity_type_path)
    relation_type_dict = load_json(relation_type_path)
    unique_relation = load_json(relation_list_path)
    current_key =load_json(checkpoint_file)
    
    return entities_dict, node_type_dict, relation_type_dict, unique_relation,  current_key


def parse_args():
    parser = argparse.ArgumentParser(description='Generate entities and relations from dataset')
    parser.add_argument('--model_config_path', default='/fred/oz479/ICT30017-P39-RAG-Watermarking-for-ownership-user-atribution/ragwm/model_configs/llama7b_config.json', type=str)
    parser.add_argument('--model_name', type=str, default='gpt3.5')
    parser.add_argument("--eval_model_code", type=str, default="contriever", choices=["contriever","contriever-msmarco","ance"])
    parser.add_argument('--eval_dataset', type=str, default='hotpotqa', help='BEIR dataset to evaluate', choices= ['trec-covid','nfcorpus','nq', 'msmarco', 'hotpotqa'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--basepath', type=str, default='/fred/oz479/ICT30017-P39-RAG-Watermarking-for-ownership-user-atribution/ragwm/output/wm_prepare')
    parser.add_argument('--dataset_prob', type=float, default=0.01, help='for nfcorpus 1')

    return parser.parse_args()


if __name__ == '__main__':
    print("generate_entity")
    args = parse_args()

   # LLM setup
    if args.model_config_path is None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'
    config = load_json(args.model_config_path)

    # Check the provider to decide if we need API keys
    provider = config.get("model_info", {}).get("provider", "").lower()

    if provider == "llama":
        # Local Llama doesn't need these, so we set them to None or skip
        api_keys = None
        api_base = None
        print("Local Llama provider detected. Skipping OpenAI API setup.")
    else:
        # Standard API setup for gpt3.5/gpt4
        api_keys = config["api_key_info"]["api_keys"][0]
        api_base = config["api_key_info"].get("api_base", [None])[0]
        os.environ["OPENAI_API_KEY"] = api_keys
        if api_base:
            os.environ["OPENAI_API_BASE"] = api_base

    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    import torch

    # This is where your Llama weights are actually parked
    # If you found them in /fred/oz479/models/llama-2-7b-chat, put that here
    model_id = "meta-llama/Llama-2-7b-chat-hf" 

    # 4-bit config to fit in the 12GB P100 memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Use the path defined in your JSON
    local_path = config["model_info"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True, torch_dtype=torch.float16, device_map="auto")
    model_id, 
    quantization_config=bnb_config,
    device_map="auto"
    

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    llm_transformer = LLMGraphTransformer(llm=llm)
    # Load dataset
    if args.eval_dataset == 'closed_qa':
        train_dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
        closed_qa_dataset = train_dataset.filter(lambda example: example['category'] == 'closed_qa')
        ndataset = data_prepare('closed_qa', closed_qa_dataset)
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
        ndataset = data_prepare(args.eval_dataset, corpus)

    basepath = os.path.join(args.basepath, args.eval_dataset)
    LOG_FILE = os.path.join(basepath,'log.log')
    file_exist(LOG_FILE)

    logger = Log(log_file=LOG_FILE).get(__file__)

    logger.info(f'args:{args}')
    print(LOG_FILE)

    generate_entity(ndataset, basepath, prob=args.dataset_prob)

 

# 'nfcorpus 3633 item, prob is set 1
# python entity_generate/generate_entity_llm_check.py --eval_dataset 'nfcorpus' --dataset_prob 1
# 'trec-covid 171332 item, prob is set 0.03 , 
# python entity_generate/generate_entity_llm_check.py --eval_dataset 'trec-covid' --dataset_prob 0.03
# total 2681468  prob is set 0.0019
# python entity_generate/generate_entity_llm_check.py  --eval_dataset 'nq' --dataset_prob 0.0019
# total 8841823  prob is set 0.00057
# python entity_generate/generate_entity_llm_check.py --eval_dataset 'msmarco' --dataset_prob 0.00057
# total 5233329 prob is set 0.00096
# python entity_generate/generate_entity_llm_check.py  --eval_dataset  'hotpotqa'  --dataset_prob 0.00096
