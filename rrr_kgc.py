import os
import sys
import json
import re
import argparse
import datetime
from datasets import Datasets
from llm_api import run_gpt_chat, run_llm_chat
from prompt_templates import H2T_TASK_TEMPLATE, T2H_TASK_TEMPLATE, EXTRACT_TEMPLATE, RANKING_TEMPLATE
from utils import parse_answer_list_response, cal_inter_candidate_list, combine_inter_and_rest_after_extract, judge_train_val_valid, ensemble_many_lists

def run_kgc_ranking(dataset, local, forward, fs_ctx, wiki_ctx, cand_ctx, fsl, cand_num, embedding, start_idx, end_idx):
    ds = Datasets(dataset, embedding, forward)
    entity2detail = ds.entity2detail
    alignment = ds.alignment
    e2idx, r2idx, idx2e, idx2r = ds.e2idx, ds.r2idx, ds.idx2e, ds.idx2r
    train_set = ds.train_set
    candidate_dict = ds.candidate_dict
    train_valid_set_tail_mapping = ds.train_valid_set_tail_mapping
    test_lines = ds.test_lines
    test_set_tail_mapping = ds.test_set_tail_mapping
    
    test_lines = test_lines[start_idx:end_idx] 
    embedding_list = []
    extract_list = []
    rerank_list = []
    ensemble_list = []
    count = start_idx
    for line in test_lines:
        print("*"*50)
        print(f"Line: {count}")
        head_id, relation_raw, tail_id = line.strip().split('\t')
        groundtruth_id = tail_id if forward else head_id
        groundtruth = entity2detail[groundtruth_id]['label']
        head_idx, tail_idx= e2idx[head_id], e2idx[tail_id]
        relation_idx = r2idx[relation_raw]
        relation = alignment[relation_raw]
        embedding_candidate_idx_list = candidate_dict[f"{head_idx}\t{relation_idx}"] if forward else candidate_dict[f"{tail_idx}\t{relation_idx}"]
        embedding_candidate_id_list = [idx2e[idx] for idx in embedding_candidate_idx_list]   
        embedding_candidate_list = [entity2detail[candidate_id]['label'] for candidate_id in embedding_candidate_id_list]
        
        drop_filter_candidates_id = test_set_tail_mapping[f"{head_id}\t{relation_raw}"] if forward else test_set_tail_mapping[f"{tail_id}\t{relation_raw}"]
        drop_filter_candidates = [entity2detail[candidate_id]['label'] for candidate_id in drop_filter_candidates_id]
        print(f"{len(drop_filter_candidates) - 1} filter candidates to be dropped except itself: {drop_filter_candidates}")
        
        sentence = ds.load_sentence(head=entity2detail[head_id]['label'], relation_raw=relation_raw, relation=relation, tail=entity2detail[tail_id]['label'])
        
        example_messages = []
        if fsl > 0:
            cur_entity = head_id if forward else tail_id
            few_shot_pairs = ds.load_few_shot(cur_entity=cur_entity, relation=relation_raw, count=fsl)[::-1] # Reverse
            for head_e_id, tail_e_id in few_shot_pairs:
                user_message, assistant_message = "", ""
                fewshot_sentence = ds.load_sentence(head=entity2detail[head_e_id]['label'], relation_raw=relation_raw, relation=relation, tail=entity2detail[tail_e_id]['label'])
                if fs_ctx:
                    user_message += f"{entity2detail[head_e_id]['label']}: {entity2detail[head_e_id]['description']}\n" if forward else f"{entity2detail[tail_e_id]['label']}: {entity2detail[tail_e_id]['description']}\n"
                task = H2T_TASK_TEMPLATE.format(head=entity2detail[head_e_id]['label'], relation=relation, sentence=fewshot_sentence) if forward else T2H_TASK_TEMPLATE.format(tail=entity2detail[tail_e_id]['label'], relation=relation, sentence=fewshot_sentence)
                user_message += task
                answer = entity2detail[tail_e_id]['label'] if forward else entity2detail[head_e_id]['label']
                assistant_message += f"The answer is {answer}, so the [MASK] is {answer}."
                if fs_ctx:
                    assistant_message += f"\n{entity2detail[tail_e_id]['label']}: {entity2detail[tail_e_id]['description']}" if forward else f"\n{entity2detail[head_e_id]['label']}: {entity2detail[head_e_id]['description']}"
                example_messages.append({"role": "user", "content": user_message})
                example_messages.append({"role": "assistant", "content": assistant_message})
        
        task = H2T_TASK_TEMPLATE.format(head=entity2detail[head_id]['label'], relation=relation, sentence=sentence) if forward else T2H_TASK_TEMPLATE.format(tail=entity2detail[tail_id]['label'], relation=relation, sentence=sentence)

        extract_messages = example_messages.copy()
        extract_candidate_id_list = []
        if wiki_ctx:
            # Reasoning stage is preprocessed and saved. If you want to test the reasoning stage where no context is provided, you should activate the json file in no_ctx version.
            gpt_extract_file = f"./data/{dataset}/reasoning_preprocessed_forward.json" if forward else f"./data/{dataset}/reasoning_preprocessed_backward.json"
            # gpt_extract_file = f"./data/{dataset}/reasoning_preprocessed_forward_no_ctx.json" if forward else f"./data/{dataset}/reasoning_preprocessed_backward_no_ctx.json"
            if os.path.exists(gpt_extract_file):
                with open(gpt_extract_file, 'r', encoding='utf-8') as f:
                    gpt_extract_response = json.load(f)
                extract_response = gpt_extract_response[f"{head_id}\t{relation_raw}"] if forward else gpt_extract_response[f"{tail_id}\t{relation_raw}"]
            else:
                materials = f"{entity2detail[head_id]['label']}: {entity2detail[head_id]['wikipedia_intro_long']}" if forward else f"{entity2detail[tail_id]['label']}: {entity2detail[tail_id]['wikipedia_intro_long']}"
                extract_prompt = EXTRACT_TEMPLATE.format(materials=materials, task=task)
                extract_messages.append({"role": "user", "content": extract_prompt})
                for m in extract_messages:
                    print(m)
                extract_response = run_llm_chat(extract_messages, forward) if local else run_gpt_chat(extract_messages)
            print("LLM extract_response:", extract_response)
            llm_candidate_list = parse_answer_list_response(extract_response)
            inter_candidate_id_list, rest_embedding_candidate_id_list = cal_inter_candidate_list(llm_candidate_list, embedding_candidate_id_list, entity2detail)
            inter_candidate_list = [entity2detail[candidate_id]['label'] for candidate_id in inter_candidate_id_list]
            extract_candidate_id_list, new_cand_num = combine_inter_and_rest_after_extract(inter_candidate_id_list, embedding_candidate_id_list, rest_embedding_candidate_id_list, cand_num)
            extract_candidate_list = [entity2detail[candidate_id]['label'] for candidate_id in extract_candidate_id_list]

            print("LLM Candidate List:", llm_candidate_list)
            print("Inter Candidate List:", inter_candidate_list)
            print(f"{embedding} Candidate List:", embedding_candidate_list)
            print("Extract Candidate List:", extract_candidate_list)
            rerank_candidate_list = extract_candidate_list[:new_cand_num]
            rerank_candidate_id_list = extract_candidate_id_list[:new_cand_num]
            print(f"{new_cand_num} Candidate List for Rerank: {rerank_candidate_list}")
        else:
            rerank_candidate_list = embedding_candidate_list[:cand_num]
            rerank_candidate_id_list = embedding_candidate_id_list[:cand_num]
            print(f"{cand_num} Candidate List for Rerank: {rerank_candidate_list}")

        candidate_list_ctx = ""
        if cand_ctx:
            for candidate_id in rerank_candidate_id_list:
                candidate_list_ctx += f"\n{entity2detail[candidate_id]['label']}: {entity2detail[candidate_id]['description']}" # 短context
            candidate_list_ctx += "\n"

        ranking_messages = example_messages.copy()
        ranking_prompt = RANKING_TEMPLATE.format(candidate_list=', '.join(rerank_candidate_list), task=task, cadidate_list_ctx=candidate_list_ctx)
        if cand_ctx:
            ranking_prompt = f'{entity2detail[head_id]["label"]}: {entity2detail[head_id]["description"]}\n{ranking_prompt}' if forward else f'{entity2detail[tail_id]["label"]}: {entity2detail[tail_id]["description"]}\n{ranking_prompt}'
        ranking_messages.append({"role": "user", "content": ranking_prompt})
        print(f"Ranking Prompt:{ranking_messages[-1]}")
        
        response = run_llm_chat(ranking_messages, forward) if local else run_gpt_chat(ranking_messages)
        print("LLM Response:", response) 
        response_list = parse_answer_list_response(response)

        inter_candidate_id_list, rest_embedding_candidate_id_list = cal_inter_candidate_list(response_list, embedding_candidate_id_list, entity2detail)
        inter_candidate_id_list = [candidate_id for candidate_id in inter_candidate_id_list if judge_train_val_valid(entity2detail[candidate_id]['label'], head_id, relation_raw, tail_id, forward, train_valid_set_tail_mapping)]
        inter_candidate_list = [entity2detail[candidate_id]['label'] for candidate_id in inter_candidate_id_list]
        rest_embedding_candidate_list = [entity2detail[candidate_id]['label'] for candidate_id in rest_embedding_candidate_id_list]
        final_candidate_id_list = inter_candidate_id_list + rest_embedding_candidate_id_list
        final_candidate_list = inter_candidate_list + rest_embedding_candidate_list
        print(f"Final Candidate List: {final_candidate_list}")

        embedding_candidate_id_list = [candidate_id for candidate_id in embedding_candidate_id_list if (candidate_id not in drop_filter_candidates_id or candidate_id == groundtruth_id)]
        extract_candidate_id_list = [candidate_id for candidate_id in extract_candidate_id_list if (candidate_id not in drop_filter_candidates_id or candidate_id == groundtruth_id)]
        final_candidate_id_list = [candidate_id for candidate_id in final_candidate_id_list if (candidate_id not in drop_filter_candidates_id or candidate_id == groundtruth_id)]

        filtered_final_candate_list = [entity2detail[candidate_id]['label'] for candidate_id in final_candidate_id_list]
        print(f"Final Candidate List after filtering: {filtered_final_candate_list}")
        print(f"Groundtruth: {groundtruth} ID: {groundtruth_id}")
        
        embedding_hits = embedding_candidate_id_list.index(groundtruth_id) + 1 if groundtruth_id in embedding_candidate_id_list else 0
        print(f"{embedding} Hits@{embedding_hits}")
        embedding_list.append(embedding_hits)
        
        extract_hits = extract_candidate_id_list.index(groundtruth_id) + 1 if groundtruth_id in extract_candidate_id_list else 0
        print(f"Extract Hits@{extract_hits}")
        extract_list.append(extract_hits)

        rerank_hits = final_candidate_id_list.index(groundtruth_id) + 1 if groundtruth_id in final_candidate_id_list else 0
        print(f"Rerank Hits@{rerank_hits}")
        rerank_list.append(rerank_hits)
        
        if count % 100 == 0:
            print(f"{embedding} Result: MRR: {sum(1/h for h in embedding_list if h >= 1) / len(embedding_list):.3f} Hits@1: {sum(1 for h in embedding_list if h == 1) / len(embedding_list):.3f} Hits@3: {sum(1 for h in embedding_list if h <= 3 and h >= 1) / len(embedding_list):.3f} Hits@10: {sum(1 for h in embedding_list if h <= 10 and h >= 1) / len(embedding_list):.3f}")
            print(f"Extract Result: MRR: {sum(1/h for h in extract_list if h >= 1) / len(extract_list):.3f} Hits@1: {sum(1 for h in extract_list if h == 1) / len(extract_list):.3f} Hits@3: {sum(1 for h in extract_list if h <= 3 and h >= 1) / len(extract_list):.3f} Hits@10: {sum(1 for h in extract_list if h <= 10 and h >= 1) / len(extract_list):.3f}")
            print(f"Rerank Result: MRR: {sum(1/h for h in rerank_list if h >= 1) / len(rerank_list):.3f} Hits@1: {sum(1 for h in rerank_list if h == 1) / len(rerank_list):.3f} Hits@3: {sum(1 for h in rerank_list if h <= 3 and h >= 1) / len(rerank_list):.3f} Hits@10: {sum(1 for h in rerank_list if h <= 10 and h >= 1) / len(rerank_list):.3f}")
        count += 1        
    
    print("Final Result:")
    print(f"{embedding} Result: MRR: {sum(1/h for h in embedding_list if h >= 1) / len(embedding_list):.3f} Hits@1: {sum(1 for h in embedding_list if h == 1) / len(embedding_list):.3f} Hits@3: {sum(1 for h in embedding_list if h <= 3 and h >= 1) / len(embedding_list):.3f} Hits@10: {sum(1 for h in embedding_list if h <= 10 and h >= 1) / len(embedding_list):.3f}")
    print(f"Extract Result: MRR: {sum(1/h for h in extract_list if h >= 1) / len(extract_list):.3f} Hits@1: {sum(1 for h in extract_list if h == 1) / len(extract_list):.3f} Hits@3: {sum(1 for h in extract_list if h <= 3 and h >= 1) / len(extract_list):.3f} Hits@10: {sum(1 for h in extract_list if h <= 10 and h >= 1) / len(extract_list):.3f}")
    print(f"Rerank Result: MRR: {sum(1/h for h in rerank_list if h >= 1) / len(rerank_list):.3f} Hits@1: {sum(1 for h in rerank_list if h == 1) / len(rerank_list):.3f} Hits@3: {sum(1 for h in rerank_list if h <= 3 and h >= 1) / len(rerank_list):.3f} Hits@10: {sum(1 for h in rerank_list if h <= 10 and h >= 1) / len(rerank_list):.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Knowledge Graph Completion with various modes.")
    parser.add_argument('--forward', action='store_true', help='forward KGC')
    parser.add_argument('--local', action='store_true', help='Local LLM mode') # GPT default if you do not add --local
    parser.add_argument('--dataset', type=str, default='FB15k237', choices=['FB15k237', 'YAGO3-10'], help='dataset name')
    parser.add_argument('--fs_ctx', action='store_true', help='context in fewshot or not')
    parser.add_argument('--wiki_ctx', action='store_true', help='context from wikipedia or not')
    parser.add_argument('--cand_ctx', action='store_true', help='context in candidate list or not')
    parser.add_argument('--fsl', type=int, default=2, help='Few-shot learning mode')
    parser.add_argument('--cand_num', type=int, default=5, help='candidate number')
    parser.add_argument('--embedding', type=str, default='RotatE', help='embedding mode')
    parser.add_argument('--start_idx', type=int, default=9, help='start index of test set')
    parser.add_argument('--end_idx', type=int, default=10, help='end index of test set')
    args = parser.parse_args()

    if args.end_idx == 0:
        args.end_idx = None
    
    run_kgc_ranking(dataset=args.dataset, forward=args.forward, local=args.local, fs_ctx=args.fs_ctx, wiki_ctx=args.wiki_ctx, cand_ctx=args.cand_ctx, fsl=args.fsl, cand_num=args.cand_num, embedding=args.embedding, start_idx=args.start_idx, end_idx=args.end_idx)
    