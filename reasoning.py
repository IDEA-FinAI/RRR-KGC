import os
import sys
import json
import re
import argparse
import datetime

# from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

API_KEY = "" # your OpenAI API key

H2T_TASK_TEMPLATE = "The question is to predict the tail entity [MASK] from the given ({head}, {relation}, [MASK]) by completing the sentence '{sentence}'."
T2H_TASK_TEMPLATE = "The question is to predict the head entity [MASK] from the given ([MASK], {relation}, {tail}) by completing the sentence '{sentence}'."

EXTRACT_TEMPLATE = "Here are some materials for you to refer to. \n{materials}\n\
{task} \
Output all the possible answers you can find in the materials using the format [answer1, answer2, ..., answerN] and please start your response with 'The possible answers:'. \
Do not output anything except the possible answers. If you cannot find any answer, please output some possible answers based on your own knowledge."

EXTRACT_TEMPLATE_WITHOUT_CTX = "{task}\nOutput some possible answers based on your knowledge using the format [answer1, answer2, ..., answerN] and please start your response with 'The possible answers:'. \
Do not output anything except the possible answers."


device = "cuda"

def run_llm_chat(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        do_sample=False,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


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
    response = response.replace('The possible answers: \n- ', '').replace('\n-', ',')
    return response.replace('The possible answers:', '').replace('The final order:', '').replace('[', '').replace(']', '').strip().split(', ')

def run_kgc_ranking(dataset, model_name, forward, fs_ctx, wiki_ctx, fsl, start_idx, end_idx, model_path):
    entity2wiki = load_entity2wikidata(f'./data/{dataset}/entity2detail.json')
    alignment = load_alignment(f'./data/{dataset}/alignment.txt')
    e2idx, r2idx, idx2e, idx2r = load_entity_relation(dataset)

    if model_path is None:
        model_path = model_name

    if 'GPT' not in model_name:
        # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # tokenizer.pad_token = tokenizer.eos_token
        # generation_config = dict(
        #     temperature=0,
        #     top_k=0,
        #     top_p=0,
        #     do_sample=False,
        #     max_new_tokens=512,
        # )
        # model.cuda()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

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

    for _, line in enumerate(tqdm(test_lines)):
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
            if forward:
                entity_intro = entity2wiki[head_id]['wikipedia_intro_long'] if 'wikipedia_intro_long' in entity2wiki[head_id]['wikipedia_intro_long'] else None
                if entity_intro is None and 'wikipedia_intro' in entity2wiki[head_id]:
                    entity_intro = entity2wiki[head_id]['wikipedia_intro']
                if entity_intro is None and 'description' in entity2wiki[head_id]:
                    entity_intro = entity2wiki[head_id]['description']
                if entity_intro is None:
                    entity_intro = entity2wiki[head_id]['label']
            else:
                entity_intro = entity2wiki[tail_id]['wikipedia_intro_long'] if 'wikipedia_intro_long' in \
                                                                               entity2wiki[tail_id][
                                                                                   'wikipedia_intro_long'] else None
                if entity_intro is None and 'wikipedia_intro' in entity2wiki[tail_id]:
                    entity_intro = entity2wiki[tail_id]['wikipedia_intro']
                if entity_intro is None and 'description' in entity2wiki[tail_id]:
                    entity_intro = entity2wiki[tail_id]['description']
                if entity_intro is None:
                    entity_intro = entity2wiki[tail_id]['label']
            materials = f"{entity2wiki[head_id]['label']}: {entity_intro}" if forward else f"{entity2wiki[tail_id]['label']}: {entity_intro}"
            extract_prompt = EXTRACT_TEMPLATE.format(materials=materials, task=task)
            # Reasoning without context
            #   extract_prompt = EXTRACT_TEMPLATE_WITHOUT_CTX.format(task=task)
            extract_messages.append({"role": "user", "content": extract_prompt})
            if "GPT" in model_name:
                extract_response = run_gpt_chat(extract_messages)
            else:
                extract_response = run_llm_chat(model, tokenizer, extract_messages)
                # print(extract_response)
                # print(extract_messages)
                # print('\n')
            llm_response_dict[key] = extract_response
            os.makedirs("extract", exist_ok=True)
            direction = "forward" if forward else "backward"
            with open(f"extract/{model_name}_extract_{dataset}_{direction}.json", 'w', encoding='utf-8') as file:
                json.dump(llm_response_dict, file, indent=4, ensure_ascii=False)
        else:
            extract_prompt = EXTRACT_TEMPLATE_WITHOUT_CTX.format(task=task)
            extract_messages.append({"role": "user", "content": extract_prompt})
            if "GPT" in model_name:
                extract_response = run_gpt_chat(extract_messages)
            else:
                extract_response = run_llm_chat(model, tokenizer, extract_messages)
            llm_response_dict[key] = extract_response
            os.makedirs("extract", exist_ok=True)
            direction = "forward" if forward else "backward"
            with open(f"extract/{model_name}_extract_{dataset}_{direction}_no_ctx.json", 'w', encoding='utf-8') as file:
                json.dump(llm_response_dict, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Knowledge Graph Completion with various modes.")
    parser.add_argument('--gen', action='store_true', help='Generative KGC')
    parser.add_argument('--rank', action='store_true', help='Ranking KGC')
    parser.add_argument('--forward', action='store_true', help='forward KGC')
    #   parser.add_argument('--local', action='store_true', help='Local LLM mode') # GPT default
    parser.add_argument('--model_name', type=str, default='GPT', )
    parser.add_argument('--model_path', type=str, default='/data/FinAi_Mapping_Knowledge/finllm/LLMs/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='FB15k237', choices=['FB15k237', 'WN18RR', 'YAGO3-10'], help='dataset name')
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
        run_kgc_ranking(dataset=args.dataset, forward=args.forward, model_name=args.model_name, fs_ctx=args.fs_ctx,
                        wiki_ctx=args.wiki_ctx, fsl=args.fsl, start_idx=args.start_idx, end_idx=args.end_idx, model_path=args.model_path)
    