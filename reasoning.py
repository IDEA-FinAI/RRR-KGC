import os
import sys
import json
import re
import argparse
import datetime
from openai import OpenAI

API_KEY = "" # your OpenAI API key

H2T_TASK_TEMPLATE = "The question is to predict the tail entity [MASK] from the given ({head}, {relation}, [MASK]) by completing the sentence '{sentence}'."
T2H_TASK_TEMPLATE = "The question is to predict the head entity [MASK] from the given ([MASK], {relation}, {tail}) by completing the sentence '{sentence}'."

# EXTRACT_TEMPLATE = "Here are some materials for you to refer to. \n{materials}\n\
# {task} \
# Output all the possible answers you can find in the materials using the format [answer1, answer2, ..., answerN] and please start your response with 'The possible answers:'. \
# Do not output anything except the possible answers. If you cannot find any answer, please output some possible answers based on your own knowledge."

EXTRACT_TEMPLATE = "{task}\nOutput some possible answers based on your knowledge using the format [answer1, answer2, ..., answerN] and please start your response with 'The possible answers:'. \
Do not output anything except the possible answers."

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

def messages_to_prompt(messages):
    return '\n'.join([message['content'] for message in messages])

def load_alignment(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        alignment = json.load(file)
    return alignment

def load_entity2wikidata(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        entity2wikidata = json.load(file)
    return entity2wikidata

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
    with open(f'./data/{dataset}/entities.txt', 'r') as entity_file:
        entities = entity_file.read().splitlines()
        e2idx = {entity: idx for idx, entity in enumerate(entities)}
        idx2e = {idx: entity for idx, entity in enumerate(entities)}

    with open(f'./data/{dataset}/relations.txt', 'r') as relation_file:
        relations = relation_file.read().splitlines()
        r2idx = {relation: idx for idx, relation in enumerate(relations)}
        idx2r = {idx: relation for idx, relation in enumerate(relations)}

    return e2idx, r2idx, idx2e, idx2r

def parse_answer_list_response(response):
    return response.replace('The possible answers:', '').replace('The final order:', '').replace('[', '').replace(']', '').strip().split(', ')

def run_kgc_ranking(dataset, local, forward, fs_ctx, wiki_ctx, fsl, start_idx, end_idx):
    entity2wiki = load_entity2wikidata(f'./data/{dataset}/entity2detail.json')
    alignment = load_alignment(f'./data/{dataset}/alignment.txt')
    e2idx, r2idx, idx2e, idx2r = load_entity_relation(dataset)

    train_set = {}
    with open(f'data/{dataset}/train.txt', 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            head, relation, tail = parts[0], parts[1], parts[2]
            if relation not in train_set: 
                train_set[relation] = []
            train_set[relation].append((head, tail))
    
    with open(f'data/{dataset}/test.txt', 'r', encoding='utf-8') as f:
        test_lines = f.readlines()
    
    test_lines = test_lines[start_idx:end_idx] 
    count = start_idx
    llm_response_dict = {}

    for line in test_lines:
        head_id, relation_raw, tail_id = line.strip().split('\t')
        groundtruth_id = tail_id if forward else head_id
        groundtruth = entity2wiki[groundtruth_id]['label']
        head_idx, tail_idx= e2idx[head_id], e2idx[tail_id]
        relation_idx = r2idx[relation_raw]
        relation = alignment[relation_raw]
        
        key = f"{head_id}\t{relation_raw}" if forward else f"{tail_id}\t{relation_raw}"
        if key in llm_response_dict:
            continue

        sentence = load_sentence(head=entity2wiki[head_id]['label'], relation_raw=relation_raw, relation=relation, tail=entity2wiki[tail_id]['label'], forward=forward, dataset=dataset)
        
        example_messages = []
        if fsl > 0:
            cur_entity = head_id if forward else tail_id
            few_shot_pairs = load_few_shot(cur_entity, relation_raw, train_set, fsl, forward)[::-1] # Reverese
            for head_e_id, tail_e_id in few_shot_pairs:
                user_message, assistant_message = "", ""
                fewshot_sentence = load_sentence(head=entity2wiki[head_e_id]['label'], relation_raw=relation_raw, relation=relation, tail=entity2wiki[tail_e_id]['label'], forward=forward, dataset=dataset)
                if fs_ctx:
                    user_message += f"{entity2wiki[head_e_id]['label']}: {entity2wiki[head_e_id]['description']}\n" if forward else f"{entity2wiki[tail_e_id]['label']}: {entity2wiki[tail_e_id]['description']}\n"
                task = H2T_TASK_TEMPLATE.format(head=entity2wiki[head_e_id]['label'], relation=relation, sentence=fewshot_sentence) if forward else T2H_TASK_TEMPLATE.format(tail=entity2wiki[tail_e_id]['label'], relation=relation, sentence=fewshot_sentence)
                user_message += task
                answer = entity2wiki[tail_e_id]['label'] if forward else entity2wiki[head_e_id]['label']
                assistant_message += f"The answer is {answer}, so the [MASK] is {answer}."
                if fs_ctx:
                    assistant_message += f"\n{entity2wiki[tail_e_id]['label']}: {entity2wiki[tail_e_id]['description']}" if forward else f"\n{entity2wiki[head_e_id]['label']}: {entity2wiki[head_e_id]['description']}"
                example_messages.append({"role": "user", "content": user_message})
                example_messages.append({"role": "assistant", "content": assistant_message})

        task = H2T_TASK_TEMPLATE.format(head=entity2wiki[head_id]['label'], relation=relation, sentence=sentence) if forward else T2H_TASK_TEMPLATE.format(tail=entity2wiki[tail_id]['label'], relation=relation, sentence=sentence)

        extract_messages = example_messages.copy()
        extract_candidate_id_list = []
        if wiki_ctx:
            # Reasoning with context
            # materials = f"{entity2wiki[head_id]['label']}: {entity2wiki[head_id]['wikipedia_intro_long']}" if forward else f"{entity2wiki[tail_id]['label']}: {entity2wiki[tail_id]['wikipedia_intro_long']}"
            # extract_prompt = EXTRACT_TEMPLATE.format(materials=materials, task=task)
            # Reasoning without context
            extract_prompt = EXTRACT_TEMPLATE.format(task=task)
            extract_messages.append({"role": "user", "content": extract_prompt})
            extract_response = run_llm_chat(extract_messages) if local else run_gpt_chat(extract_messages)
        llm_response_dict[key] = extract_response
        os.makedirs("extract", exist_ok=True)
        direction = "forward" if forward else "backward"
        with open(f"extract/gpt_extract_{dataset}_{direction}.json", 'w', encoding='utf-8') as file:
            json.dump(llm_response_dict, file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Knowledge Graph Completion with various modes.")
    parser.add_argument('--gen', action='store_true', help='Generative KGC')
    parser.add_argument('--rank', action='store_true', help='Ranking KGC')
    parser.add_argument('--forward', action='store_true', help='forward KGC')
    parser.add_argument('--local', action='store_true', help='Local LLM mode') # GPT default
    parser.add_argument('--dataset', type=str, default='FB15k237', choices=['FB15k237', 'YAGO3-10'], help='dataset name')
    parser.add_argument('--fs_ctx', action='store_true', help='context in fewshot or not')
    parser.add_argument('--wiki_ctx', action='store_true', help='context from wikipedia or not')
    parser.add_argument('--fsl', type=int, default=0, help='Few-shot learning mode')
    parser.add_argument('--start_idx', type=int, default=0, help='start index of test set')
    parser.add_argument('--end_idx', type=int, default=100, help='end index of test set')
    args = parser.parse_args()

    if args.end_idx == 0:
        args.end_idx = None
    
    if args.gen:
        run_kgc_gen(dataset=args.dataset, forward=args.forward, local=args.local, fs_ctx=args.fs_ctx, wiki_ctx=args.wiki_ctx, fsl=args.fsl, size=args.size)
    elif args.rank:
        run_kgc_ranking(dataset=args.dataset, forward=args.forward, local=args.local, fs_ctx=args.fs_ctx, wiki_ctx=args.wiki_ctx, fsl=args.fsl, start_idx=args.start_idx, end_idx=args.end_idx)
    