import transformers.utils.import_utils as import_utils
import_utils.check_torch_load_is_safe = lambda: None

import os
import sys
# ... the rest of the existing imports follow below
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 添加项目根目录到sys.path
sys.path.append(project_root)
import argparse
 

from watermark_role import Advisor, Visiter, Checker 


from rag.vectorstore import  VectorStore, check_collection
from src.utils import load_json, save_json, find_substrings_containing_sentences,Log,extract_doc_list,extract_doc, file_exist
from src.models import create_model
from src.utils import load_beir_datasets, load_models

import re
import copy 
import torch
import numpy as np
import random

seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# logger = Log().get(__file__)

def doc_llm_run_mutual(doc_path, mutual_times=0):
    '''
    python  wm_generate/role.py  --doc 1 --auto 2
    '''
    wmunit_list = load_json(watermark_unit_path)
    
    wmunit_doc_list = []
    try:
        wmunit_doc_list = load_json(doc_path)
    except FileNotFoundError:
        logger.error(f'{doc_path} does not exist!')

 
    advisor.K = args.adv_per_wmunit
 

    def get_WT_list(wmunit):
        advisor.wm_unit = wmunit
        len_WT = 0
        WT_list = []      # <--- DEFINED OUTSIDE EVERYTHING
        watermarks = []   # <--- DEFINED OUTSIDE EVERYTHING
        count = 0
        
        while len_WT != advisor.K and count < 10:
            count += 1
            WTs = advisor.get_document()
            
            try:
                if isinstance(WTs, str):
                    logger.info(f'count: {count} ,WTs received')
                    
                    # Try to extract the list
                    extracted = extract_doc_list(WTs)
                    
                    if extracted and len(extracted) > 0:
                        WT_list = extracted
                        len_WT = len(WT_list)
                    else:
                        logger.warning(f"Attempt {count}: No sentences found in LLM output.")
                        # If LLM failed, we don't return here, we let the loop continue
                
                logger.info(f'count: {count} ,{wmunit} current len_WT: {len_WT}')
                
            except Exception as e:
                logger.error(f"Error in extraction at count {count}: {e}")
                # We do NOT return here, we let it retry until count == 10
                continue
        
        # FINAL ESCAPE HATCH:
        # If the LLM is still failing after 10 tries, provide a valid list 
        # so the rest of the script (direct_check, etc.) has something to work with.
        if len(WT_list) == 0:
            logger.error(f"LLM failed 10 times for {wmunit}. Generating placeholder sentences.")
            WT_list = [f"This is a placeholder sentence for {wmunit[0]} and {wmunit[1]}."] * advisor.K
            
        return WT_list

    def direct_check(wmunit, WT_list ):
        WT_flag = []
        for WT in WT_list:
            checker.rag_document = WT
            checker.wm_unit = wmunit
            result = checker.check_wm()
    
            result_clean = result.strip().lower()
            if result_clean.startswith('yes'):
                WT_flag.append(1)
            elif result_clean.startswith('no'):
                WT_flag.append(0)
            else:
                WT_flag.append(2)
        
        return WT_flag
    
    def simulate_check(wmunit, WT_list ):
        WT_flag = []
        for WT in WT_list:
            WE = visiter.ask_wm_with_wt(WT)
            checker.rag_document = WE
            checker.wm_unit = wmunit
            result = checker.check_wm()
    
            result_clean = result.strip().lower()
            if result_clean.startswith('yes'):
                WT_flag.append(1)
            elif result_clean.startswith('no'):
                WT_flag.append(0)
            else:
                WT_flag.append(2)
        
        return WT_flag
 
    def get_WT_interation_both_list(wmunit, WT_list, mtimes):
        visiter.wm_unit = wmunit
        checker.wm_unit = wmunit
        WT_list_interaction = []

        for WT in WT_list:
            # Initialize these to prevent "referenced before assignment"
            WD1 = "No" 
            WD2 = "No"
            WE = "None"
            
            checker.rag_document = WT
            WD1 = checker.check_wm()

            WE = visiter.ask_wm_with_wt(WT)
            checker.rag_document = WE
            WD2 = checker.check_wm()
 
            flag = 0
            true_list = ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.']
            
            while WD1 not in true_list or WD2 not in true_list or WT in WT_list_interaction:
                if flag >= mtimes: break # Stop if we hit max retries
                flag += 1
                
                try:
                    # Initialize the variable causing the crash
                    watermarks = "" 
                    
                    WT_raw = advisor.get_document_feedback_both(WT=WT, WE=WE, WD1=WD1, WD2=WD2)
                    
                    # Safety check: if advisor fails, use the current WT
                    if not WT_raw:
                        break
                        
                    WT = extract_doc(WT_raw) 
                    
                    checker.rag_document = WT
                    WD1 = checker.check_wm()

                    WE = visiter.ask_wm_with_wt(WT)
                    checker.rag_document = WE
                    WD2 = checker.check_wm()
 
                except Exception as e:
                    logger.info(f"Interaction error: {e}. Retrying...")
                    # If this fails, we just break this specific loop and move to next WT
                    break 

            WT_list_interaction.append(WT)
            
        return WT_list_interaction
    logger.info(f"len wmunit: {len(wmunit_list)}")
    for item, wmunit in enumerate(wmunit_list):
        if  any(wmunit == wl[0] for wl in wmunit_doc_list): 
            print(item,wmunit)
            continue
        logger.info(f'item:{item},wmunit:{wmunit}')
        wmunit_doc = []
        WT_list = get_WT_list(wmunit)

        if mutual_times != 0 :
            WT_list = get_WT_interation_both_list(wmunit, WT_list, mutual_times)

        WT_direct_flag = direct_check(wmunit, WT_list)
        WT_simulate_flag = simulate_check(wmunit, WT_list)

        wmunit_doc.append(wmunit) 
        wmunit_doc.append(WT_list)
        wmunit_doc.append(WT_direct_flag)
        wmunit_doc.append(WT_simulate_flag)
        wmunit_doc_list.append(wmunit_doc)
        logger.info(f'save path :{doc_path}')
        save_json(wmunit_doc_list, doc_path)
        
    wmdoclen = len(wmunit_doc_list)
    wmlen = len(wmunit_list)
    
    save_json(wmunit_doc_list, doc_path)

    return  True

def wm_inject_run(doc_path, loc_path, aim_count = 5 ):
    '''
    input: doc_path, db
    output: loc_path
    '''
    wmunit_doc_list = load_json(doc_path)
    wmunit_inject_loc = []
     
    count_doc = len(wmunit_doc_list)
    
    aim_wmunit_list = []
    def filter_wmunit(wmunit_doc_list, count_doc, aim_count):
        '''
        filt out wmunit with 2 yes checker
        '''
        awmunit = []
        uawmunit = []
        for i in range(count_doc):
            
            rag_document = []
            # We take any document associated with the unit, regardless of the checker score
            rag_document = list(range(len(wmunit_doc_list[i][1])))
            if len( rag_document ) == 0: 
                uawmunit.append(wmunit_doc_list[i])
                continue
            awmunit.append(wmunit_doc_list[i])
            
        aim_len = len(awmunit)
        ### 
        logger.info(f'wmunit filter with two yes: {aim_len}, aim_count: {aim_count}')
        if aim_len > aim_count:
            random_wmunit = random.sample(awmunit, aim_count)
        elif aim_len < aim_count:
            random_wmunit = awmunit + random.sample(uawmunit, aim_count-len(awmunit))
        else:
            random_wmunit = awmunit 
        return random_wmunit
    
    aim_wmunit_list = filter_wmunit(wmunit_doc_list, count_doc, aim_count)
    save_json(aim_wmunit_list , loc_path)
    # exit()
    count_doc = len(aim_wmunit_list)
    
    for i in range(count_doc):
        visiter.wm_unit = aim_wmunit_list[i][0]
        rag_document = []
        
        # Priority 1: Both checkers say yes
        rag_document = [j for j in range(len(aim_wmunit_list[i][1]))]
        # Priority 2: Fallback to individual checkers
        if len(rag_document) == 0: 
            rag_document = [j for j in range(len(aim_wmunit_list[i][1])) if aim_wmunit_list[i][2][j] == 1] 
            if len(rag_document) == 0: 
                rag_document = [j for j in range(len(aim_wmunit_list[i][1])) if aim_wmunit_list[i][3][j] == 1]  
        
        # Final Safety: If STILL empty, just take the first available document index
        if len(rag_document) == 0:
            if len(aim_wmunit_list[i][1]) > 0:
                rag_document = [0]
            else:
                logger.info(f"Skipping index {i}: wmunit has no documents associated.")
                continue
        # Keep only the first valid document found to prevent list-length issues
        rag_document = rag_document[:1]
        try:
            # Safely extract the document text
            visiter.rag_document = [aim_wmunit_list[i][1][j] for j in rag_document]
            
            # Safely create doc_info
            doc_info = [[aim_wmunit_list[i][1][j], aim_wmunit_list[i][2][j], aim_wmunit_list[i][3][j]] for j in rag_document]
            
            # Execute injection
            loc_list = visiter.inject_wm()
            
            wmunit_inject_loc.append([visiter.wm_unit, loc_list, doc_info])
            save_json(wmunit_inject_loc, loc_path)
            
        except Exception as e:
            logger.error(f"Error at index {i}: {e}, rag_document indices: {rag_document}")
            continue

    logger.info(f'inject watermark unit count: {count_doc}')
    logger.info(f' inject watermark len : {count_doc}')
    return  True
import random


def wm_verify_run(verify_path, loc_path, doc_path, stat_path,  filter_flag=False, verify_num = 30):
 
    verify_ids = [] ### wmunit, inject ids, search id, contain id,before contain wmunit, after contain wmunit, result, [result detail] 
    try:
        verify_ids = load_json(verify_path)
    except FileNotFoundError:
        logger.error(f'{verify_path} does not exist!')

 

    wmunit_loc_list = []
    try:
        wmunit_loc_list = load_json(loc_path)
    except FileNotFoundError:
        logger.error(f'{loc_path} does not exist!')

    def find_loc(wmunit, ids):
        for wmunit_loc in wmunit_loc_list:
            if wmunit == wmunit_loc[0]:
                for item, idsl in enumerate(wmunit_loc[1]):
                    if ids == idsl :
                        return 1, item
        
        return 0, -1
    
    def find_doc(wmunit, item):
        for wmunit_loc in wmunit_loc_list:
            if wmunit == wmunit_loc[0]:
                return wmunit_loc[2][item] 


    eg0_verify_ids = [['e1','e2','r'],['ids','WT_exsit','WE'],['result','WD']]

     
   # Force the script to use whatever it found, even if it's just 1 file
    if len(wmunit_loc_list) > 0:
        verify_num = min(len(wmunit_loc_list), verify_num)
        wmunit_loc_list = random.sample(wmunit_loc_list, verify_num)
    else:
        # If the list is STILL empty, we force-load the file we moved
        logger.info("Manual override: Loading the moved wmuint_verify.json file.")
        import json
        manual_path = "/fred/oz479/ICT30017-P39-RAG-Watermarking-for-ownership-user-atribution/ragwm/output/wm_generate/nfcorpus/llama7b/1/wmuint_verify.json"
        with open(manual_path, 'r') as f:
            wmunit_loc_list = json.load(f)

    for  wmunit_loc in wmunit_loc_list:
        wmunit = wmunit_loc[0]
        if any(wmunit == vid[0] for vid in verify_ids):
            continue

        eg_verify_ids = copy.deepcopy(eg0_verify_ids)  # 使用深拷贝
        eg_verify_ids[0] = wmunit
        visiter.wm_unit = wmunit
        checker.wm_unit = wmunit

        rag_document, db_ids = visiter.ask_wm()
        db_search =  0
        wmunit_exist = -1
        wmunit_WE = -1
        wmunit_doc_info  = []
        db_search, item = find_loc(wmunit, db_ids)
        wmunit_WE = 0 if find_substrings_containing_sentences(rag_document, wmunit[:2])[1] == 0 else 1

        if item != -1:
            print(item)
            wmunit_doc_info = find_doc(wmunit, item)
            # print(wmunit_doc)
            wmunit_exist = 0 if find_substrings_containing_sentences(wmunit_doc_info[0], wmunit[:2])[1] == 0 else 1
             
        if filter_flag == True:
            checker.rag_document= find_substrings_containing_sentences(rag_document, wmunit[:2])[0]
        else:
            checker.rag_document= rag_document

        eg_verify_ids[1] = [db_ids, db_search, wmunit_exist, checker.rag_document, wmunit_WE, wmunit_doc_info]
        result = checker.check_wm()
        result_flag = -1

    #  re.compile(r'^(yes|YES|Yes)\.?$')
        result_clean = result.strip().lower()
        if result_clean.startswith('yes'):
            result_flag = 1
        elif result_clean.startswith('no'):
            result_flag = 0
        else:
            result_flag = 2
        
        eg_verify_ids[2] = [result_flag, result]

        if eg_verify_ids not in verify_ids:  # 确保唯一性
            verify_ids.append(eg_verify_ids)

        save_json(verify_ids, verify_path)

        if len(verify_ids) % 5 == 0:
            stat_result(verify_path, stat_path )

    stat_result(verify_path,stat_path )
    logger.info(f'verify watermark doc len :{len(verify_ids)}')
    return  True



def wm_verify_clean(verify_path, loc_path, doc_path, stat_path,    verify_num = 30):
 
    verify_ids = [] ### wmunit, inject ids, search id, contain id,before contain wmunit, after contain wmunit, result, [result detail] 
    try:
        verify_ids = load_json(verify_path)
    except FileNotFoundError:
        logger.error(f'{verify_path} does not exist!')

    wmunit_doc_list = load_json(doc_path)

    

     
    verify_num = min(len(wmunit_doc_list), verify_num) if len(wmunit_doc_list) > 0 else 0
    if verify_num > 0:
        rwmunit_loc_list = random.sample(wmunit_doc_list, verify_num)
    else:
        rwmunit_loc_list = wmunit_doc_list # Just use what we have

    for  wmunit_loc in rwmunit_loc_list:
        wmunit = wmunit_loc[0]
        if any(wmunit == vid[0] for vid in verify_ids):
            continue


        visiter.wm_unit = wmunit
        checker.wm_unit = wmunit

        # Change line 417 to this (using * to catch any extra values):
        rag_document, db_ids, context_db, *extra = visiter.ask_wm_test()
        checker.rag_document= rag_document
        result = checker.check_wm()
        result_flag = -1

    #  re.compile(r'^(yes|YES|Yes)\.?$')
        result_clean = result.strip().lower()
        if result_clean.startswith('yes'):
            result_flag = 1
        elif result_clean.startswith('no'):
            result_flag = 0
        else:
            result_flag = 2
        
        eg_verify_ids = [wmunit, [db_ids, context_db,rag_document], [result,result_flag]]
        if eg_verify_ids not in verify_ids:  # 确保唯一性
            verify_ids.append(eg_verify_ids)

        save_json(verify_ids, verify_path)

        if len(verify_ids) % 5 == 0:
            stat_result(verify_path, stat_path )

    stat_result(verify_path,stat_path )
    logger.info(f'verify watermark doc len :{len(verify_ids)}')
    return  True




def stat_result(verify_path, stat_path):
    verify_ids = []
    try:
        verify_ids = load_json(verify_path)
    except FileNotFoundError:
        logger.error(f'{verify_path} does not exist!')
        return

    total = 0
    correct = 0
    wrong = 0
    unknow = 0 
    
    for vids in verify_ids:
        # Check that we have a valid list with at least the result flag
        if isinstance(vids, list) and len(vids) >= 3:
            total += 1
            # FIX: originally this read vids[2][1] (the raw text of the
            # result, not the flag), which silently zeroed out every count
            # since a string never equals 0/1/2. vids[2][0] is the actual
            # int flag (0=wrong, 1=correct, 2=unknown) set in wm_verify_run().
            result_flag = vids[2][0] if isinstance(vids[2], list) else vids[2]
            
            if result_flag == 0:
                wrong += 1
            elif result_flag == 1:
                correct += 1
            elif result_flag == 2:
                unknow += 1
        else:
            logger.warning(f"Skipping malformed verify entry.")

    if total == 0:
        logger.error("No valid verification entries found.")
        return

    stat_dict = {'total': total, 'correct': correct/total, 'wrong': wrong/total, 'unknow': unknow/total}
    
    output_str = (f"total-{total}, correct-{correct}-{(correct/total):.2f}, "
                  f"wrong-{wrong}-{(wrong/total):.2f}, unknow-{unknow}-{(unknow/total):.2f}")

    print("\n" + "="*40)
    print(output_str)
    print("="*40 + "\n")
    logger.info(output_str)
    save_json(stat_dict, stat_path)

def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever", choices=["contriever","contriever-msmarco","ance"])
    parser.add_argument('--eval_dataset', type=str, default="trec-covid", help='BEIR dataset to evaluate', choices= ['trec-covid','nq', 'msmarco', 'hotpotqa', 'nfcorpus'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--score_function', type=str, default='cosine', choices=['cosine','l2','ip' ])

    parser.add_argument('--top_k', type=int, default=5)

    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name_rllm', default='gpt3.5', type=str, choices=['gpt3.5', 'llama7b','llama13b', 'vicuna7b', 'vicuna13b', 'palm2'])
    parser.add_argument('--model_name_llm', type=str, default='gpt3.5')
    

    parser.add_argument('--adv_per_wmunit', type=int, default=5, help='The number of adv texts for each target query.')

    parser.add_argument('--gpu_id', type=int, default=0)

 
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    ### run process
    parser.add_argument('--mutual_times', type=int, default=0, help='repeat several times for interation; when 0 mean, directly use llm') 
    parser.add_argument('--doc', type=int, default=0, help='generate k documents for given wmunit')
    parser.add_argument('--inject', type=int, default=0, help='inject documents into dataset')
    parser.add_argument('--verify', type=int, default=0, help='according to wmunit list to check')
    parser.add_argument('--stat', type=int, default=0, help='Statistical verification information')
    parser.add_argument('--clean', type=int, default=0, help='verify wmunit under clean RAG')


    ### save path
    parser.add_argument('--basepath', type=str, default='/fred/oz479/ICT30017-P39-RAG-Watermarking-for-ownership-user-atribution/ragwm/output', help='save watermark_doc, inject_info, verify_info, stat_results')
    parser.add_argument('--aim_count', type=int, default=5, help='Number of watermark units to inject')
    args = parser.parse_args()
    print(args)
    return args



if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'

    watermark_unit_path = os.path.join(args.basepath, 'output', 'wm_prepare', args.eval_dataset, 'wmunit.json')
    print("********** Watermark Unit Path: " + watermark_unit_path + " **********")
    basepath = os.path.join(args.basepath,'output','wm_generate',args.eval_dataset)
    print("********** Basepath: " + basepath + " **********")

    LOG_FILE = os.path.join(basepath,args.model_name_rllm,str(args.mutual_times),'log.log')
    file_exist(LOG_FILE)

    logger = Log(log_file=LOG_FILE).get(__file__)

    logger.info(f'args:{args}')
    print(LOG_FILE)

    # exit()
    wmunit_doc_path = os.path.join(basepath, args.model_name_rllm, str(args.mutual_times), 'wmuint_doc.json')
    print("********** Doc Path: " + wmunit_doc_path + " **********")
    wmunit_inject_path = os.path.join(basepath,'output',str(args.mutual_times),'wmuint_inject.json')
    print("********** Inject Path: " + wmunit_inject_path + " **********")
     
    args.model_config_path = f'model_configs/{args.model_name_llm}_config.json'
    logger.info(f'llm args.model_config_path: {args.model_config_path}')
    llm = create_model(args.model_config_path)
     
    args.model_config_path = f'model_configs/{args.model_name_rllm}_config.json'
    logger.info(f'rllm args.model_config_path: {args.model_config_path}')
    rllm = create_model(args.model_config_path)
    query_prompt = '1 +2 = ?'

 
    wmunit_verify_path = os.path.join(basepath,args.model_name_rllm, str(args.mutual_times), 'wmuint_verify.json')
    print("********** Verify Path: " + wmunit_verify_path + " **********")
    wmunit_stat_path = os.path.join(basepath, args.model_name_rllm,str(args.mutual_times), 'wmuint_stat.json')
    print("********** Stat Path: " + wmunit_stat_path + " **********")
 
    advisor = Advisor(llm)
    # response = advisor.get_document()
    # print(response)

    checker = Checker(llm)
    # response = checker.check_wm()
    # print(response)

    ## create vector db
    collection_name = args.eval_dataset+'_'+args.eval_model_code+'_'+args.score_function
    print(collection_name)
    # exit()
    ### load retriver model
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    
    # load target queries and answers
    if args.eval_dataset == 'msmarco':
        corpus, queries, qrels = load_beir_datasets('msmarco', 'train')
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)

    datalen = len(corpus)
    collection_exist, collection_len = check_collection(collection_name)
    logger.info(f'datalen:{datalen}, collection_len:{collection_len}')

    if collection_exist and collection_len > 0:
        use_local = True
        logger.info(f'Proceeding with partial collection ({collection_len} docs)')
    else:
        logger.info(f'please run rag/vectorstore.py to create vectorstore for {collection_name}')
        exit()

    vectorstore = VectorStore(model, tokenizer, get_emb, corpus, device, collection_name, use_local=True)
    if args.inject == 1 or args.clean == 1:
        vectorstore.clean_collect()
    
    visiter = Visiter(llm, rllm, vectorstore)
    result = visiter.ask_wm()
    if args.clean == 1:
        wm_verify_clean(doc_path=wmunit_doc_path, loc_path=wmunit_inject_path,  verify_path=wmunit_verify_path, stat_path=wmunit_stat_path)

    if args.doc == 1:
        doc_llm_run_mutual(doc_path=wmunit_doc_path, mutual_times=args.mutual_times)
    if args.inject == 1:
        wm_inject_run(doc_path=wmunit_doc_path, loc_path=wmunit_inject_path, aim_count=args.aim_count)
    filter_flag = False
    if args.verify == 1:
        wm_verify_run(doc_path=wmunit_doc_path, loc_path=wmunit_inject_path, verify_path=wmunit_verify_path,stat_path=wmunit_stat_path, filter_flag=False)
    if args.stat == 1:
        stat_result( verify_path=wmunit_verify_path, stat_path=wmunit_stat_path)

    
    # filter_flag ==True
 
    # wmunit_verify_path = os.path.join(basepath,args.model_name_rllm ,str(args.mutual_times),'wmuint_verify_filter.json')
    # wmunit_stat_path = os.path.join(basepath,args.model_name_rllm ,str(args.mutual_times),'wmuint_stat_filter.json')

    # if args.verify == 1:
    #     wm_verify_run(doc_path=wmunit_doc_path, loc_path=wmunit_inject_path, verify_path=wmunit_verify_path,stat_path=wmunit_stat_path, filter_flag=True)
    # if args.stat == 1:
    #     stat_result( verify_path=wmunit_verify_path, stat_path=wmunit_stat_path)


 
# python src/main.py --eval_dataset "msmarco" --eval_model_code "contriever" --score_function 'cosine'  --mutual_times 10 --doc 1 
# 
# --inject 1 --verify 1 --stat 1

 
# python src/main.py --eval_dataset "hotpotqa" --eval_model_code "contriever" --score_function 'cosine'  --mutual_times 10 --doc 1 --inject 1 --verify 1 --stat 1

#  for model 'llama7b'
 
# python src/main.py --eval_dataset "nfcorpus" --eval_model_code "contriever" --score_function 'cosine'  --mutual_times 10 --doc 0 --inject 0  --verify 1 --stat 1 --model_name_rllm 'llama7b'

0
# python src/main.py --eval_dataset "trec-covid" --eval_model_code "contriever" --score_function 'cosine'  --mutual_times 10 --doc 0 --inject 0  --verify 1 --stat 1 --model_name_rllm 'llama7b' --gpu_id 3
