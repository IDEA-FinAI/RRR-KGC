import os
import sys
import json
import re
import argparse
import datetime
from openai import OpenAI

API_KEY = "" # Your API_KEY Here

H2T_TASK_TEMPLATE = "The question is to predict the tail entity [MASK] from the given ({head}, {relation}, [MASK]) by completing the sentence '{sentence}'."
T2H_TASK_TEMPLATE = "The question is to predict the head entity [MASK] from the given ([MASK], {relation}, {tail}) by completing the sentence '{sentence}'."

EXTRACT_TEMPLATE = "Here are some materials for you to refer to. \n{materials}\n\
{task} \
Output all the possible answers you can find in the materials using the format '[answer1 | answer2 | ... | answerN]' and please start your response with 'The possible answers:'. \
Do not output anything except the possible answers. If you cannot find any answer, please output some possible answers based on your own knowledge."

RANKING_TEMPLATE = "{task} \
The list of candidate answers is [{candidate_list}]. \
{cadidate_list_ctx}\
Sort the list to let the candidate answers which are more possible to be the true answer to the question prior. \
Output the sorted order of candidate answers using the format '[most possible answer, second possible answer, ..., least possible answer]' and please start your response with 'The final order:'. "

def run_gpt_chat(messages):
    client = OpenAI(
        api_key=API_KEY,
    )
    attempt = 0
    while attempt < 3:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                top_p=0,
                messages=messages,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print("GPT Generate Error:", e)
            attempt += 1
    return "Error"

def run_llm_chat(messages, forward):
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1" if forward else "http://localhost:8001/v1" # Your Local LLM API Port Here
    )
    attempt = 0
    while attempt < 3:
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3-8B-Instruct",
                temperature=0,
                top_p=0,
                messages=messages,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print("LLM Generate Error:", e)
            attempt += 1
    return "Error"

def messages_to_prompt(messages):
    return '\n'.join([message['content'] for message in messages])

def load_few_shot(cur_entity, relation, train_set, count, forward):
    few_shot_pairs = []

    if relation in train_set:
        for h, t in train_set[relation]:
            if (forward and h == cur_entity) or (not forward and t == cur_entity):
                few_shot_pairs.append((h, t))
                if len(few_shot_pairs) >= count:
                    return few_shot_pairs

        for h, t in train_set[relation]:
            if (h, t) not in few_shot_pairs:
                few_shot_pairs.append((h, t))
                if len(few_shot_pairs) >= count:
                    return few_shot_pairs

    return few_shot_pairs

def load_sentence(head, relation_raw, relation, tail, forward, dataset='FB15k237'):
    if dataset == 'FB15k237':
        last_relation = relation_raw.split('/')[-1]
        first_property = relation_raw.split('/')[2]
        if forward:
            return f'what is the {last_relation} of {first_property} {head}? The answer is '
        else:
            return f'{tail} is the {last_relation} of what {first_property}? The answer is '
    elif dataset == 'YAGO3-10':
        if forward:
            return f'{head} {relation} what? The answer is '
        else:
            return f'what {relation} {tail}? The answer is '

def load_entity_relation(dataset: str = 'FB15k237'):
    with open(f'./data/{dataset}/entities.txt', 'r', encoding='utf-8') as entity_file:
        entities = entity_file.read().splitlines()
        e2idx = {entity: idx for idx, entity in enumerate(entities)}
        idx2e = {idx: entity for idx, entity in enumerate(entities)}

    with open(f'./data/{dataset}/relations.txt', 'r', encoding='utf-8') as relation_file:
        relations = relation_file.read().splitlines()
        r2idx = {relation: idx for idx, relation in enumerate(relations)}
        idx2r = {idx: relation for idx, relation in enumerate(relations)}

    return e2idx, r2idx, idx2e, idx2r

def parse_answer_list_response(response):
    return response.replace('The possible answers:', '').replace('The final order:', '').replace('[', '').replace(']', '').strip().split(', ')

def cal_inter_candidate_list(llm_candidate_list, embedding_candidate_id_list, entity2detail):
    inter_candidate_id_list = []
    embedding_candidate_id_seen = set()

    for candidate in llm_candidate_list:
        for embedding_candidate_id in embedding_candidate_id_list:
            if candidate == entity2detail[embedding_candidate_id]['label'] or candidate in entity2detail[embedding_candidate_id]['alternatives']:
                if embedding_candidate_id not in embedding_candidate_id_seen:
                    inter_candidate_id_list.append(embedding_candidate_id)
                    embedding_candidate_id_seen.add(embedding_candidate_id)
                break

    rest_embedding_candidate_id_list = [candidate_id for candidate_id in embedding_candidate_id_list if candidate_id not in inter_candidate_id_list]
    return inter_candidate_id_list, rest_embedding_candidate_id_list

def combine_inter_and_rest_after_extract(inter_candidate_id_list, embedding_candidate_id_list, rest_embedding_candidate_id_list, cand_num):
    list_for_rerank = embedding_candidate_id_list[:cand_num]
    list_rest = embedding_candidate_id_list[cand_num:]
    
    for inter_id in inter_candidate_id_list:
        if inter_id not in list_for_rerank:
            list_for_rerank.append(inter_id)

    list_rest = [candidate_id for candidate_id in list_rest if candidate_id not in inter_candidate_id_list]
    
    rerank_size = len(list_for_rerank)
    final_list = list_for_rerank + list_rest
    return final_list, rerank_size

def judge_train_val_valid(entity_label, head_id, relation_raw, tail_id, forward, train_valid_set_tail_mapping):
    key = f"{head_id}\t{relation_raw}" if forward else f"{tail_id}\t{relation_raw}"
    if key in train_valid_set_tail_mapping:
        if entity_label in train_valid_set_tail_mapping[key]:
            return False
    return True

def ensemble_many_lists(embedding_candidate_id_list, final_candidate_id_list):
    scores = {}
    for idx, candidate in enumerate(embedding_candidate_id_list, start=1):
        scores[candidate] = scores.get(candidate, 0) + 1 / idx
    for idx, candidate in enumerate(final_candidate_id_list, start=1):
        scores[candidate] = scores.get(candidate, 0) + 1 / idx

    sorted_candidates = sorted(scores, key=scores.get, reverse=True)
    return sorted_candidates

def run_kgc_ranking(dataset, local, forward, fs_ctx, wiki_ctx, cand_ctx, fsl, cand_num, embedding, start_idx, end_idx):
    with open(f'./data/{dataset}/entity2detail.json', 'r', encoding='utf-8') as file:
        entity2detail = json.load(file)
    with open(f'./data/{dataset}/alignment.txt', 'r', encoding='utf-8') as file:
        alignment = json.load(file)
    e2idx, r2idx, idx2e, idx2r = load_entity_relation(dataset)

    train_set = {}
    with open(f'data/{dataset}/train.txt', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            head, relation, tail = parts[0], parts[1], parts[2]
            if relation not in train_set:
                train_set[relation] = []
            train_set[relation].append((head, tail))

    retriever_file = f'./data/{dataset}/{embedding}_retriever_candidate_tail.txt' if forward else f'./data/{dataset}/{embedding}_retriever_candidate_head.txt'
    
    with open(retriever_file, 'r', encoding='utf-8') as f:
        candidate_dict = json.load(f)
    
    with open(f'data/{dataset}/train.txt', 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    
    with open(f'data/{dataset}/valid.txt', 'r', encoding='utf-8') as f:
        valid_lines = f.readlines()
        
    train_valid_lines = train_lines + valid_lines
    train_valid_set_tail_mapping = {}
    for line in train_valid_lines:
        head_id, relation_raw, tail_id = line.strip().split('\t')
        key = f"{head_id}\t{relation_raw}" if forward else f"{tail_id}\t{relation_raw}"
        if key not in train_valid_set_tail_mapping:
            train_valid_set_tail_mapping[key] = set()
        head_label = entity2detail[head_id]['label']
        tail_label = entity2detail[tail_id]['label']
        train_valid_set_tail_mapping[key].add(tail_label) if forward else train_valid_set_tail_mapping[key].add(head_label)
        
    with open(f'data/{dataset}/test.txt', 'r', encoding='utf-8') as f:
        test_lines = f.readlines()

    test_set_tail_mapping = {}
    for line in test_lines:
        head_id, relation_raw, tail_id = line.strip().split('\t')
        key = f"{head_id}\t{relation_raw}" if forward else f"{tail_id}\t{relation_raw}"
        if key not in test_set_tail_mapping:
            test_set_tail_mapping[key] = set()
        test_set_tail_mapping[key].add(tail_id) if forward else test_set_tail_mapping[key].add(head_id)
    
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
        
        sentence = load_sentence(head=entity2detail[head_id]['label'], relation_raw=relation_raw, relation=relation, tail=entity2detail[tail_id]['label'], forward=forward, dataset=dataset)
        
        example_messages = []
        if fsl > 0:
            cur_entity = head_id if forward else tail_id
            few_shot_pairs = load_few_shot(cur_entity, relation_raw, train_set, fsl, forward)[::-1] # Reverse
            for head_e_id, tail_e_id in few_shot_pairs:
                user_message, assistant_message = "", ""
                fewshot_sentence = load_sentence(head=entity2detail[head_e_id]['label'], relation_raw=relation_raw, relation=relation, tail=entity2detail[tail_e_id]['label'], forward=forward, dataset=dataset)
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
            gpt_extract_file = f"./data/{dataset}/gpt_extract_wiki_forward.json" if forward else f"./data/{dataset}/gpt_extract_wiki_backward.json"
            # gpt_extract_file = f"./data/{dataset}/gpt_extract_wiki_forward_no_ctx.json" if forward else f"./data/{dataset}/gpt_extract_wiki_backward_no_ctx.json" # reasoning without ctx
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
    parser.add_argument('--gen', action='store_true', help='Generative KGC')
    parser.add_argument('--rank', action='store_true', help='Ranking KGC')
    parser.add_argument('--forward', action='store_true', help='forward KGC')
    parser.add_argument('--local', action='store_true', help='Local LLM mode') # GPT default
    parser.add_argument('--dataset', type=str, default='FB15k237', choices=['FB15k237', 'YAGO3-10'], help='dataset name')
    parser.add_argument('--fs_ctx', action='store_true', help='context in fewshot or not')
    parser.add_argument('--wiki_ctx', action='store_true', help='context from wikipedia or not')
    parser.add_argument('--cand_ctx', action='store_true', help='context in candidate list or not')
    parser.add_argument('--fsl', type=int, default=0, help='Few-shot learning mode')
    parser.add_argument('--cand_num', type=int, default=10, help='candidate number')
    parser.add_argument('--embedding', type=str, default='GIE', help='embedding mode')
    parser.add_argument('--start_idx', type=int, default=0, help='start index of test set')
    parser.add_argument('--end_idx', type=int, default=100, help='end index of test set')
    args = parser.parse_args()

    if args.end_idx == 0:
        args.end_idx = None
    
    if args.gen:
        run_kgc_gen(dataset=args.dataset, forward=args.forward, local=args.local, fs_ctx=args.fs_ctx, wiki_ctx=args.wiki_ctx, cand_ctx=args.cand_ctx, fsl=args.fsl, size=args.size)
    elif args.rank:
        run_kgc_ranking(dataset=args.dataset, forward=args.forward, local=args.local, fs_ctx=args.fs_ctx, wiki_ctx=args.wiki_ctx, cand_ctx=args.cand_ctx, fsl=args.fsl, cand_num=args.cand_num, embedding=args.embedding, start_idx=args.start_idx, end_idx=args.end_idx)
    