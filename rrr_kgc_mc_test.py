import argparse
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_manager import DataManager
from datetime import datetime


def load_triples(dataset, subset):
    triples = []
    with open('./data/{}/{}.txt'.format(dataset, subset), mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            triples.append(line.strip().split('\t'))
    return triples


def load_entity2wikidata(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        entity2wikidata = json.load(file)
    return entity2wikidata


def cal_Y_prob(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, generation_config, prompt_list):
    messages_batch = [
        [{"role": "user", "content": prompt}]
        for prompt in prompt_list
    ]
    texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in
             messages_batch]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    generated_output = model.generate(
        input_ids=inputs.input_ids,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        **generation_config
    )

    scores = generated_output.scores[0]
    probs = scores.softmax(dim=-1)

    Y_id = tokenizer.encode("Y", add_special_tokens=False)[0]
    N_id = tokenizer.encode("N", add_special_tokens=False)[0]

    Y_probs = [probs[i, Y_id].item() for i in range(probs.shape[0])]
    N_probs = [probs[i, N_id].item() for i in range(probs.shape[0])]

    final_probs =[probs[i, Y_id].item() / (probs[i, Y_id].item() + probs[i, N_id].item()) for i in range(probs.shape[0])]
    #   return Y_probs
    return final_probs


def log_results(fp, hits1, hits3, hits10, mrr):
    fp.write(f"Hits@1: {hits1}\n")
    fp.write(f"Hits@3: {hits3}\n")
    fp.write(f"Hits@10: {hits10}\n")
    fp.write(f"MRR: {mrr}\n")


def main(dataset, kge, llm, start, end, llm_batch_size, model_path, reasoning, use_reasoning_ctx, use_ctx):
    forward_queries = []
    backward_queries = []
    fwd_json_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-forward.json'.format(dataset, kge, llm, start, end)
    bwd_json_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-backward.json'.format(dataset, kge, llm, start, end)
    if reasoning is False and use_ctx is False:
        fwd_json_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-forward_no_reasoning_no_ctx.json'.format(dataset,
            kge, llm, start, end)
        bwd_json_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-backward_no_reasoning_no_ctx.json'.format(dataset,
            kge, llm, start, end)
    if reasoning is False and use_ctx is True:
        fwd_json_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-forward_no_reasoning.json'.format(dataset, kge,
                                                                                                  llm, start, end)
        bwd_json_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-backward_no_reasoning.json'.format(dataset, kge,
                                                                                                   llm, start, end)
    if reasoning is True and use_ctx is False and use_reasoning_ctx is True:
        fwd_json_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-forward_no_ctx.json'.format(dataset, kge, llm, start, end)
        bwd_json_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-backward_no_ctx.json'.format(dataset, kge, llm, start, end)
    if reasoning is True and use_ctx is False and use_reasoning_ctx is False:
        fwd_json_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-forward_no_reasoning_ctx_no_ctx.json'.format(dataset,
            kge, llm, start, end)
        bwd_json_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-backward_no_reasoning_ctx_no_ctx.json'.format(dataset,
            kge, llm, start, end)
    if reasoning is True and use_ctx is True and use_reasoning_ctx is False:
        fwd_json_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-forward_no_reasoning_ctx.json'.format(dataset, kge, llm,
            start, end)
        bwd_json_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-backward_no_reasoning_ctx.json'.format(dataset, kge, llm,
            start, end)
    print('Reasoning is {}, use context is {}'.format(reasoning, use_ctx))
    with open(fwd_json_file_name, mode='r', encoding='utf-8') as f:
        forward_queries = json.load(f)
    with open(bwd_json_file_name, mode='r',
              encoding='utf-8') as f:
        backward_queries = json.load(f)

    entity2wiki = load_entity2wikidata('./data/{}/entity2wikidata.json'.format(dataset))

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    generation_config = dict(
        temperature=0,
        top_k=0,
        top_p=0,
        do_sample=False,
        max_new_tokens=10,
    )
    model.cuda()

    train_triples = load_triples(dataset, 'train')
    valid_triples = load_triples(dataset, 'valid')
    test_triples = load_triples(dataset, 'test')
    existing_tails_for_hr = {}
    existing_heads_for_rt = {}
    for triple_set in [train_triples, valid_triples, test_triples]:
        for tp in triple_set:
            hr = (tp[0], tp[1])
            rt = (tp[1], tp[2])
            head = tp[0]
            tail = tp[2]
            if hr in existing_tails_for_hr:
                existing_tails_for_hr[hr].add(tail)
            else:
                existing_tails_for_hr[hr] = set()
                existing_tails_for_hr[hr].add(tail)
            if rt in existing_heads_for_rt:
                existing_heads_for_rt[rt].add(head)
            else:
                existing_heads_for_rt[rt] = set()
                existing_heads_for_rt[rt].add(head)

    with open('./logs/{}-{}-{}-{}-fwd.log'.format(dataset, kge, start, end), mode='w', encoding='utf-8') as f:
        f.write('Forward KGC test on dataset {} with backbone embedding model {}'.format(dataset, kge))
        fwd_hits1 = []
        fwd_hits3 = []
        fwd_hits10 = []
        fwd_reasoning_only_hits1 = []
        fwd_reasoning_only_hits3 = []
        fwd_reasoning_only_hits10 = []
        fwd_reassoning_only_mrr = []
        fwd_mrr = []
        for idx, query in enumerate(tqdm(forward_queries)):
            triple = query['triple']
            triple_with_labels = query['seq_triple']
            mode = query['mode']
            gt = query['gt']
            instruction = query['instructions']
            rerank_scope = query['rerank_scope']
            init_ranking = query['init_ranking']
            init_ranking_with_label = query['init_ranking_label']

            if 'Qwen2' in model_path:
                messages = [{"role": "user", "content": instruction}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to("cuda")
                generated_ids = model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    do_sample=False,
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]

                pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                messages_batch = [
                    [{"role": "user", "content": instruction}]
                ]
                texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for
                         messages in
                         messages_batch]
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

                generated_output = model.generate(
                    input_ids=inputs.input_ids,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **generation_config
                )
                output = generated_output.sequences[0].cpu().numpy().tolist()
                pred = tokenizer.decode(output, skip_special_tokens=True).strip()
                if 'Llama-2' in model_path:
                    pred = pred.split('[/INST] ')[-1].strip()
                else:
                    pred = pred.split('assistant')[-1].strip()
            if gt not in init_ranking:
                fwd_hits1.append(0)
                fwd_hits3.append(0)
                fwd_hits10.append(0)
                fwd_mrr.append(0)
                fwd_reasoning_only_hits1.append(0)
                fwd_reasoning_only_hits3.append(0)
                fwd_reasoning_only_hits10.append(0)
                fwd_reassoning_only_mrr.append(0)
                continue
            #   perform filtering
            filtered_samples = existing_tails_for_hr[(triple[0], triple[1])]
            init_ranking = [e for e in init_ranking if e not in filtered_samples or e == gt]
            init_ranking_with_label = [entity2wiki[e]['label'] for e in init_ranking]
            raw_rank = init_ranking.index(gt) + 1
            gt = entity2wiki[gt]['label']
            rank = raw_rank
            if gt == pred:
                rank = 1
            else:
                if pred not in set(init_ranking_with_label) or init_ranking_with_label.index(pred) >= raw_rank:
                    rank = raw_rank + 1

            hits1 = 1 if rank <= 1 else 0
            hits3 = 1 if rank <= 3 else 0
            hits10 = 1 if rank <= 10 else 0
            fwd_hits1.append(hits1)
            fwd_hits3.append(hits3)
            fwd_hits10.append(hits10)
            fwd_mrr.append(float(1/rank))
            reasoning_only_rank = raw_rank
            reasoning_only_hits1 = 1 if reasoning_only_rank <= 1 else 0
            reasoning_only_hits3 = 1 if reasoning_only_rank <= 3 else 0
            reasoning_only_hits10 = 1 if reasoning_only_rank <= 10 else 0
            fwd_reasoning_only_hits1.append(reasoning_only_hits1)
            fwd_reasoning_only_hits3.append(reasoning_only_hits3)
            fwd_reasoning_only_hits10.append(reasoning_only_hits10)
            fwd_reassoning_only_mrr.append(1/reasoning_only_rank)
            if (idx + 1) % 100 == 0:
                avg_fwd_hits1 = round(float(sum(fwd_hits1) / len(fwd_hits1)), 3)
                avg_fwd_hits3 = round(float(sum(fwd_hits3) / len(fwd_hits3)), 3)
                avg_fwd_hits10 = round(float(sum(fwd_hits10) / len(fwd_hits10)), 3)
                avg_fwd_mrr = round(float(sum(fwd_mrr) / len(fwd_mrr)), 3)
                f.write(f"\nMetrics after processing {idx + 1} batches:\n")
                log_results(f, avg_fwd_hits1, avg_fwd_hits3, avg_fwd_hits10, avg_fwd_mrr)
                f.write("\n" + "=" * 50 + "\n")
        avg_fwd_hits1 = round(float(sum(fwd_hits1) / len(fwd_hits1)), 3)
        avg_fwd_hits3 = round(float(sum(fwd_hits3) / len(fwd_hits3)), 3)
        avg_fwd_hits10 = round(float(sum(fwd_hits10) / len(fwd_hits10)), 3)
        avg_fwd_mrr = round(float(sum(fwd_mrr) / len(fwd_mrr)), 3)
        f.write(f"\nFinal forward prediction performance after processing {idx + 1} batches:\n")
        log_results(f, fwd_hits1, fwd_hits3, fwd_hits10, fwd_mrr)
        f.write("\n" + "=" * 50 + "\n")
    print('Final forward prediction performance: ')
    print(f'Hits@1 = {avg_fwd_hits1}, Hits@3 = {avg_fwd_hits3}, Hits@10 = {avg_fwd_hits10}, MRR = {avg_fwd_mrr}')

    with open('./logs/{}-{}-{}-{}-bwd.log'.format(dataset, kge, start, end), mode='w', encoding='utf-8') as f:
        bwd_hits1 = []
        bwd_hits3 = []
        bwd_hits10 = []
        bwd_mrr = []
        for idx, query in enumerate(tqdm(backward_queries)):
            triple = query['triple']
            triple_with_labels = query['seq_triple']
            mode = query['mode']
            gt = query['gt']
            instruction = query['instruction']
            rerank_scope = query['rerank_scope']
            init_ranking = query['init_ranking']
            init_ranking_with_label = query['init_ranking_label']

            if gt not in init_ranking:
                bwd_hits1.append(0)
                bwd_hits3.append(0)
                bwd_hits10.append(0)
                bwd_mrr.append(0)
                continue

            if 'Qwen2' in model_path:
                messages = [{"role": "user", "content": instruction}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to("cuda")
                generated_ids = model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    do_sample=False,
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]

                pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                messages_batch = [
                    [{"role": "user", "content": instruction}]
                ]
                texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for
                         messages in
                         messages_batch]
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

                generated_output = model.generate(
                    input_ids=inputs.input_ids,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **generation_config
                )
                output = generated_output.sequences[0].cpu().numpy().tolist()
                pred = tokenizer.decode(output, skip_special_tokens=True).strip()
                if 'Llama-2' in model_path:
                    pred = pred.split('[/INST] ')[-1].strip()
                else:
                    pred = pred.split('assistant')[-1].strip()

            filtered_samples = existing_heads_for_rt[(triple[1], triple[2])]
            init_ranking = [e for e in init_ranking if e not in filtered_samples or e == gt]
            init_ranking_with_label = [entity2wiki[e]['label'] for e in init_ranking]
            raw_rank = init_ranking.index(gt) + 1
            gt = entity2wiki[gt]['label']
            rank = raw_rank
            if gt == pred:
                rank = 1
            else:
                if pred not in set(init_ranking_with_label) or init_ranking_with_label.index(pred) >= raw_rank:
                    rank = raw_rank + 1

            hits1 = 1 if rank <= 1 else 0
            hits3 = 1 if rank <= 3 else 0
            hits10 = 1 if rank <= 10 else 0
            bwd_hits1.append(hits1)
            bwd_hits3.append(hits3)
            bwd_hits10.append(hits10)
            bwd_mrr.append(float(1 / rank))
            if (idx + 1) % 100 == 0:
                avg_bwd_hits1 = round(float(sum(bwd_hits1) / len(bwd_hits1)), 3)
                avg_bwd_hits3 = round(float(sum(bwd_hits3) / len(bwd_hits3)), 3)
                avg_bwd_hits10 = round(float(sum(bwd_hits10) / len(bwd_hits10)), 3)
                avg_bwd_mrr = round(float(sum(bwd_mrr) / len(bwd_mrr)), 3)
                f.write(f"\nMetrics after processing {idx + 1} batches:\n")
                log_results(f, avg_bwd_hits1, avg_bwd_hits3, avg_bwd_hits10, avg_bwd_mrr)
                f.write("\n" + "=" * 50 + "\n")
        avg_bwd_hits1 = round(float(sum(bwd_hits1) / len(bwd_hits1)), 3)
        avg_bwd_hits3 = round(float(sum(bwd_hits3) / len(bwd_hits3)), 3)
        avg_bwd_hits10 = round(float(sum(bwd_hits10) / len(bwd_hits10)), 3)
        avg_bwd_mrr = round(float(sum(bwd_mrr) / len(bwd_mrr)), 3)
        f.write(f"\nFinal backward prediction performance after processing {idx + 1} batches:\n")
        log_results(f, bwd_hits1, bwd_hits3, bwd_hits10, bwd_mrr)
        f.write("\n" + "=" * 50 + "\n")
    print('Final backward prediction performance: ')
    print(f'Hits@1 = {avg_bwd_hits1}, Hits@3 = {avg_bwd_hits3}, Hits@10 = {avg_bwd_hits10}, MRR = {avg_bwd_mrr}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FB15k237')
    parser.add_argument('--kge', type=str, default='NBFnet')
    parser.add_argument('--llm', type=str, default='Llama3')
    parser.add_argument('--reasoning', action='store_true', help='reasoning_part')
    parser.add_argument('--reasoning_ctx', action='store_true', help='reasoning_part')
    parser.add_argument('--use_ctx', action='store_true', help='reasoning_part')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=20466)
    parser.add_argument('--llm_batch_size', type=int, default=2)
    parser.add_argument('--model_path', type=str, default='')
    args = parser.parse_args()
    main(args.dataset, args.kge, args.llm, args.start, args.end, args.llm_batch_size, args.model_path, args.reasoning,
         args.reasoning_ctx, args.use_ctx)
