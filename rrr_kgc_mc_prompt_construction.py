import argparse
import json
import numpy as np
import os
import random


from collections import defaultdict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MC_prompt_template = f"""Here is a incomplete triple with missing {{missing_pos}} entity <missing-entity>: {{triple}}. 
Following are some contexts about {{known_pos}} entity {{entity}}:
{{entity_intro}}
Following are some triple facts of entity {{entity}}:
{{triple_facts}}
Please select the most appropriate entity for <missing-entity> from the candidate answer list: [{{candidate_answers}}].
"""

MC_prompt_template_no_ctx = f"""Here is a incomplete triple with missing {{missing_pos}} entity <missing-entity>: {{triple}}. 
Following are some triple facts of entity {{entity}}:
{{triple_facts}}
Please select the most appropriate entity for <missing-entity> from the candidate answer list: [{{candidate_answers}}]."""

MC_prompt_template_no_nb = f"""Here is a incomplete triple with missing {{missing_pos}} entity <missing-entity>: {{triple}}. 
Following are some contexts about {{known_pos}} entity {{entity}}:
{{entity_intro}}
Please select the most appropriate entity for <missing-entity> from the candidate answer list: [{{candidate_answers}}].
"""

#   You may need to adjust the model path accordingly
embedding_model = SentenceTransformer(
    model_name_or_path='models--BAAI--bge-small-en-v1.5',
    device="cuda"
)

use_nb = False

def load_triples(dataset, subset):
    triples = []
    with open('./data/{}/{}.txt'.format(dataset, subset), mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            triples.append(line.strip().split('\t'))
    return triples


def load_entity_relation(dataset):
    with open(f'./data/{dataset}/entities.txt', 'r', encoding='utf-8') as entity_file:
        entities = entity_file.read().splitlines()
        e2idx = {entity: idx for idx, entity in enumerate(entities)}
        idx2e = {idx: entity for idx, entity in enumerate(entities)}

    with open(f'./data/{dataset}/relations.txt', 'r', encoding='utf-8') as relation_file:
        relations = relation_file.read().splitlines()
        r2idx = {relation: idx for idx, relation in enumerate(relations)}
        idx2r = {idx: relation for idx, relation in enumerate(relations)}

    return e2idx, r2idx, idx2e, idx2r


def load_alignment(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        alignment = json.load(file)
    return alignment


def load_entity2wikidata(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        entity2wikidata = json.load(file)
    return entity2wikidata


def triple_to_sentence(dataset, triple, entity2wiki, relation2text):
    head, relation, tail = triple
    if dataset == "FB15k-237-subset":
        head_property = relation.split('/')[2]
        tail_property = relation.split('/')[-1]
        return f"('{entity2wiki[tail]['label']}' is the {tail_property} of {head_property} '{entity2wiki[head]['label']}')"
    elif dataset == "WN18RR-subset":
        return f"('{entity2wiki[head]['label']}' {relation2text[relation]} '{entity2wiki[tail]['label']}')"
    elif dataset == "NELL-995-subset":
        return f"('{entity2wiki[head]['label']}' {relation2text[relation]} '{entity2wiki[tail]['label']}')"
    elif dataset == "FB15k-237" or dataset =='FB15k237' or dataset == 'FB15K237':
        head_property = relation.split('/')[2]
        tail_property = relation.split('/')[-1]
        return f"('{entity2wiki[tail]['label']}' is the {tail_property} of {head_property} '{entity2wiki[head]['label']}')"
    elif dataset == 'WN18RR':
        return f"('{entity2wiki[head]['label']}' {relation2text[relation]} '{entity2wiki[tail]['label']}')"


def relation_triple_finder(test_triple, relation2headtail_dict, num_relation_triples):
    test_head, relation, test_tail = test_triple
    head_tail_pairs = relation2headtail_dict[relation]

    if len(head_tail_pairs) <= num_relation_triples:
        return [[head, relation, tail] for head, tail in head_tail_pairs if head != test_head or tail != test_tail]

    used_heads = {test_head, test_tail}
    used_tails = {test_tail, test_head}
    used_pairs = set()
    selected_triples = []
    random.shuffle(head_tail_pairs)

    for head, tail in head_tail_pairs:
        if head not in used_heads and tail not in used_tails:
            selected_triples.append([head, relation, tail])
            used_heads.add(head)
            used_tails.add(tail)
            used_pairs.add((head, tail))
            if len(selected_triples) == num_relation_triples:
                return selected_triples

    for head, tail in head_tail_pairs:
        if (head, tail) not in used_pairs:
            if len(selected_triples) < num_relation_triples:
                selected_triples.append([head, relation, tail])
                used_heads.add(head)
                used_tails.add(tail)
                used_pairs.add((head, tail))
            else:
                break
    return selected_triples


def neighbor_triple_finder(dataset, entity2wiki, relation2text, triple, entity2relationtail_dict, num_neighbor_facts):
    head, relation, tail = triple
    head_triples = entity2relationtail_dict[head]
    tail_triples = entity2relationtail_dict[tail]

    processed_head_triples = []
    for rel, t, direction in head_triples:
        if direction == 1:
            processed_head_triples.append((head, rel, t))
        else:
            processed_head_triples.append((t, rel, head))
    processed_tail_triples = []
    for rel, h, direction in tail_triples:
        if direction == 1:
            processed_tail_triples.append((tail, rel, h))
        else:
            processed_tail_triples.append((h, rel, tail))

    triple_sentence = triple_to_sentence(dataset, triple, entity2wiki, relation2text)
    # head_sentences = [
    #     triple_to_sentence((head, rel, t)) if direction == 1 else triple_to_sentence((t, rel, head))
    #     for rel, t, direction in head_triples]
    # tail_sentences = [
    #     triple_to_sentence((tail, rel, h)) if direction == 1 else self.triple_to_sentence((h, rel, tail))
    #     for rel, h, direction in tail_triples]
    head_sentences = [triple_to_sentence(dataset, tp, entity2wiki, relation2text) for tp in processed_head_triples]
    tail_sentences = [triple_to_sentence(dataset, tp, entity2wiki, relation2text) for tp in processed_tail_triples]

    all_head_sentences = [triple_sentence] + head_sentences
    all_tail_sentences = [triple_sentence] + tail_sentences

    each_count = num_neighbor_facts // 2

    top_head_sentences = head_sentences
    top_tail_sentences = tail_sentences

    top_head_triples = processed_head_triples
    top_tail_triples = processed_tail_triples
    if len(head_sentences) > each_count:
        head_embeddings = embedding_model.encode(all_head_sentences, normalize_embeddings=True)
        head_similarity = head_embeddings[0] @ head_embeddings[1:].T
        top_head_indices = np.argsort(-head_similarity)[:each_count]
        # top_head_sentences = [head_sentences[i] for i in top_head_indices]
        top_head_triples = [processed_head_triples[idx] for idx in top_head_indices]

    if len(tail_sentences) > each_count:
        tail_embeddings = embedding_model.encode(all_tail_sentences, normalize_embeddings=True)
        tail_similarity = tail_embeddings[0] @ tail_embeddings[1:].T
        top_tail_indices = np.argsort(-tail_similarity)[:each_count]
        # top_tail_sentences = [tail_sentences[i] for i in top_tail_indices]
        top_tail_triples = [processed_tail_triples[idx] for idx in top_tail_indices]

    return top_head_triples, top_tail_triples


def triple_sequentialization(triples, entity2wiki, rel2text):
    result = []
    for tp in triples:
        tp_str = '(' + entity2wiki[tp[0]]['label'] + ', ' + rel2text[tp[1]] + ', ' + entity2wiki[tp[2]]['label'] + ')'
        result.append(tp_str)
    result = '\n'.join(result)
    return result


def load_kge_rankings(candidate_dict, entity, rel, e2idx, r2idx, idx2e, idx2r):
    candidate_dict_key = '\t'.join([str(e2idx[entity]), str(r2idx[rel])])
    candidates = candidate_dict[candidate_dict_key]
    return [idx2e[c] for c in candidates]


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


def combine_inter_and_rest_after_reason(inter_candidate_id_list, embedding_candidate_id_list,
                                        rest_embedding_candidate_id_list, cand_num):
    list_for_rerank = embedding_candidate_id_list[:cand_num]
    list_rest = embedding_candidate_id_list[cand_num:]

    for inter_id in inter_candidate_id_list:
        if inter_id not in list_for_rerank:
            list_for_rerank.append(inter_id)

    list_rest = [candidate_id for candidate_id in list_rest if candidate_id not in inter_candidate_id_list]

    rerank_size = len(list_for_rerank)
    final_list = list_for_rerank + list_rest
    return final_list, rerank_size


def main(dataset, kge, llm, start, end, rerank_scope, num_rel_tps, num_nb_facts, reasoning, reasoning_ctx, use_ctx):
    train_triples = load_triples(dataset, 'train')
    valid_triples = load_triples(dataset, 'valid')
    test_triples = load_triples(dataset, 'test')
    sampled_test_triples = []
    start = start if start >= 0 else 0
    end = end if end <= len(test_triples) else len(test_triples)
    sampled_test_triples = test_triples[start:end]

    def _load_relation2headtail_dict(triple_set):
        relation2headtail_dict = defaultdict(list)
        for head, relation, tail in triple_set:
            relation2headtail_dict[relation].append([head, tail])
        return relation2headtail_dict

    def _load_entity2relationneighbor_dict(triple_set):
        entity2relationneighbor_dict = defaultdict(list)
        for head, relation, tail in triple_set:
            entity2relationneighbor_dict[head].append((relation, tail, 1))
            entity2relationneighbor_dict[tail].append((relation, head, -1))
        return entity2relationneighbor_dict

    relation2headtail_dict = _load_relation2headtail_dict(triple_set=train_triples)
    entity2relnb_dict = _load_entity2relationneighbor_dict(triple_set=train_triples)

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

    entity2wiki = load_entity2wikidata('./data/{}/entity2wikidata.json'.format(dataset))
    rel2text = load_alignment('./data/{}/alignment_naive.txt'.format(dataset))
    all_entities = list(entity2wiki.keys())

    if reasoning_ctx is True:
        reasoning_candidates_forward = {}
        with open('./data/{}/{}_reasoning_preprocessed_forward.json'.format(dataset, llm), mode='r',
                  encoding='utf-8') as f:
            reasoning_candidates_forward = json.load(f)
        reasoning_candidates_backward = {}
        with open('./data/{}/{}_reasoning_preprocessed_backward.json'.format(dataset, llm), mode='r',
                  encoding='utf-8') as f:
            reasoning_candidates_backward = json.load(f)
    else:
        reasoning_candidates_forward = {}
        with open('./data/{}/{}_reasoning_preprocessed_forward_no_ctx.json'.format(dataset, llm), mode='r',
                  encoding='utf-8') as f:
            reasoning_candidates_forward = json.load(f)
        reasoning_candidates_backward = {}
        with open('./data/{}/{}_reasoning_preprocessed_backward_no_ctx.json'.format(dataset, llm), mode='r',
                  encoding='utf-8') as f:
            reasoning_candidates_backward = json.load(f)

    #   load entity2idx and relation2idx
    e2idx, r2idx, idx2e, idx2r = load_entity_relation(dataset)

    if kge in set(('GIE', 'ComplEx', 'RotatE')):
        kge_candidates_forward = {}
        with open('./data/{}/{}_retriever_candidate_tail.txt'.format(dataset, kge), mode='r', encoding='utf-8') as f:
            kge_candidates_forward = json.load(f)
        kge_candidates_backward = {}
        with open('./data/{}/{}_retriever_candidate_head.txt'.format(dataset, kge), mode='r', encoding='utf-8') as f:
            kge_candidates_backward = json.load(f)
    elif kge in set(('NBFnet', 'SimKGC')):
        with open('./data/{}/{}_retriever_candidate.txt'.format(dataset, kge), mode='r', encoding='utf-8') as f:
            kge_candidates = json.load(f)
        kge_candidates_forward = {}
        kge_candidates_backward = {}
        for i in range(len(kge_candidates)):
            query_dict = kge_candidates[i]
            triple = query_dict['triplet']
            topk_eids = query_dict['topk_ents']
            if query_dict['inverse'] is False:
                kge_candidates_forward[tuple(triple)] = topk_eids
            else:
                kge_candidates_backward[tuple(triple)] = topk_eids
    else:
        with open('./data/{}/{}_retriever_candidate.txt'.format(dataset, kge), mode='r', encoding='utf-8') as f:
            kge_candidates = json.load(f)
        kge_candidates_forward = {}
        kge_candidates_backward = {}
        for i in range(len(kge_candidates)):
            query_dict = kge_candidates[i]
            triple = query_dict['triplet']
            topk_eids = query_dict['topk_ents']
            if query_dict['inverse'] is False:
                kge_candidates_forward[tuple(triple)] = topk_eids
            else:
                kge_candidates_backward[tuple([triple[2], triple[1], triple[0]])] = topk_eids

    bwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-backward.jsonl'.format(dataset, kge, llm, start, end)
    fwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-forward.jsonl'.format(dataset, kge, llm, start, end)
    print('Reasoning is {}, use context is {}'.format(reasoning, use_ctx))
    if reasoning is True and use_ctx is True and reasoning_ctx is True:
        bwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-backward.jsonl'.format(dataset, kge, llm, start, end)
        fwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-forward.jsonl'.format(dataset, kge, llm, start, end)
    if reasoning is True and use_ctx is True and reasoning_ctx is False:
        bwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-backward_no_reasoning_ctx.jsonl'.format(dataset, kge,
            llm, start, end)
        fwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-forward_no_reasoning_ctx.jsonl'.format(dataset, kge,
            llm, start, end)
    if reasoning is True and use_ctx is False and reasoning_ctx is True:
        bwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-backward_no_ctx.jsonl'.format(dataset,
            kge, llm, start, end)
        fwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-forward_no_ctx.jsonl'.format(dataset,
            kge, llm, start, end)
    if reasoning is True and use_ctx is False and reasoning_ctx is False:
        bwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-backward_no_reasoning_ctx_no_ctx.jsonl'.format(dataset,
            kge, llm, start, end)
        fwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-forward_no_reasoning_ctx_no_ctx.jsonl'.format(dataset,
            kge, llm, start, end)
    if reasoning is False and use_ctx is True:
        bwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-backward_no_reasoning.jsonl'.format(dataset, kge, llm,
                                                                                                    start, end)
        fwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-forward_no_reasoning.jsonl'.format(dataset, kge, llm,
                                                                                                   start, end)
    if reasoning is False and use_ctx is False:
        bwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-backward_no_reasoning_no_ctx.jsonl'.format(dataset,
                                                                                                           kge, llm,
                                                                                                           start, end)
        fwd_jsonl_file_name = '{}-{}-{}_mc_testing_prompts_{}-{}-forward_no_reasoning_no_ctx.jsonl'.format(dataset, kge,
                                                                                                          llm, start,
                                                                                                          end)

    for _, stp in enumerate(tqdm(sampled_test_triples)):
        head = stp[0]
        rel = stp[1]
        tail = stp[2]
        existing_heads = existing_heads_for_rt[(rel, tail)]
        existing_tails = existing_tails_for_hr[(head, rel)]
        reasoning_candidate_heads = reasoning_candidates_backward['\t'.join([tail, rel])]
        reasoning_candidate_tails = reasoning_candidates_forward['\t'.join([head, rel])]
        #   kge_candidate_heads and kge_candidiate_tails store entity ids such as /m/abcd01
        if kge in set(('GIE', 'ComplEx', 'RotatE')):
            kge_candidate_heads = load_kge_rankings(kge_candidates_backward, tail, rel, e2idx, r2idx, idx2e, idx2r)
            kge_candidate_tails = load_kge_rankings(kge_candidates_forward, head, rel, e2idx, r2idx, idx2e, idx2r)
        else:
            kge_candidate_heads = kge_candidates_backward[tuple(stp)]
            kge_candidate_tails = kge_candidates_forward[tuple(stp)]

        rerank_head_size = 20
        rerank_tail_size = 20
        if reasoning is True:
            reasoning_candidate_heads = parse_answer_list_response(reasoning_candidate_heads)
            reasoning_candidate_tails = parse_answer_list_response(reasoning_candidate_tails)

            intersect_head_ids, rest_head_ids = cal_inter_candidate_list(reasoning_candidate_heads, kge_candidate_heads,
                                                                         entity2wiki)
            intersect_tail_ids, rest_tail_ids = cal_inter_candidate_list(reasoning_candidate_tails, kge_candidate_tails,
                                                                         entity2wiki)
            intersect_heads = [entity2wiki[eid]['label'] for eid in intersect_head_ids]
            intersect_tails = [entity2wiki[eid]['label'] for eid in intersect_tail_ids]
            rest_heads = [entity2wiki[eid]['label'] for eid in rest_head_ids]
            rest_tails = [entity2wiki[eid]['label'] for eid in rest_tail_ids]

            rerank_head_ids, rerank_head_size = combine_inter_and_rest_after_reason(intersect_head_ids,
                                                                                    kge_candidate_heads,
                                                                                    rest_head_ids, rerank_scope)
            rerank_tail_ids, rerank_tail_size = combine_inter_and_rest_after_reason(intersect_tail_ids,
                                                                                    kge_candidate_tails,
                                                                                    rest_tail_ids, rerank_scope)
            rerank_heads = [entity2wiki[eid]['label'] for eid in rerank_head_ids]
            rerank_tails = [entity2wiki[eid]['label'] for eid in rerank_tail_ids]
        else:
            rerank_head_ids = kge_candidate_heads
            rerank_tail_ids = kge_candidate_tails
            rerank_heads = [entity2wiki[eid]['label'] for eid in kge_candidate_heads]
            rerank_tails = [entity2wiki[eid]['label'] for eid in kge_candidate_tails]

        head_nb_triples, tail_nb_triples = neighbor_triple_finder(dataset, entity2wiki, rel2text, stp,
                                                                  entity2relnb_dict, num_nb_facts)

        #   gather necessary contexts to construct prompts
        hd_label = entity2wiki[head]['label']
        hd_intro = entity2wiki[head]['wikipedia_intro'] if 'wikipedia_intro' in entity2wiki[head] else entity2wiki[head]['description']
        tl_label = entity2wiki[tail]['label']
        tl_intro = entity2wiki[tail]['wikipedia_intro'] if 'wikipedia_intro' in entity2wiki[tail] else entity2wiki[tail]['description']
        hd_triple_str = triple_sequentialization(head_nb_triples, entity2wiki, rel2text)
        tl_triple_str = triple_sequentialization(tail_nb_triples, entity2wiki, rel2text)
        fwd_test_triple_str = '(' + entity2wiki[head]['label'] + ', ' + rel2text[rel] + ', <missing-entity>)'
        bwd_test_triple_str = '(<missing-entity>, ' + rel2text[rel] + ', ' + entity2wiki[tail]['label'] + ')'

        if use_ctx is True:
            candidate_heads = []
            for ent in rerank_head_ids[:rerank_head_size]:
                if 'description' in entity2wiki[ent] and entity2wiki[ent]['description'] is not None:
                    candidate_heads.append(
                        entity2wiki[ent]['label'] + ': ' + entity2wiki[ent]['description'])
                else:
                    candidate_heads.append(
                        entity2wiki[ent]['label'] + ': ' + entity2wiki[ent]['label'])

            candidate_tails = []
            for ent in rerank_tail_ids[:rerank_tail_size]:
                if 'description' in entity2wiki[ent] and entity2wiki[ent]['description'] is not None:
                    candidate_tails.append(
                        entity2wiki[ent]['label'] + ': ' + entity2wiki[ent]['description'])
                else:
                    candidate_tails.append(
                        entity2wiki[ent]['label'] + ': ' + entity2wiki[ent]['label'])

            candidate_heads_str = '\n'.join(candidate_heads)
            candidate_tails_str = '\n'.join(candidate_tails)
            if use_nb is True:
                fwd_instruction = MC_prompt_template.format(missing_pos='tail', triple=fwd_test_triple_str,
                                                            known_pos='head',
                                                            entity=hd_label, entity_intro=hd_intro,
                                                            triple_facts=hd_triple_str,
                                                            candidate_answers=candidate_tails_str)
                bwd_instruction = MC_prompt_template.format(missing_pos='head', triple=bwd_test_triple_str,
                                                            known_pos='tail',
                                                            entity=tl_label, entity_intro=tl_intro,
                                                            triple_facts=tl_triple_str,
                                                            candidate_answers=candidate_heads_str)
            else:
                fwd_instruction = MC_prompt_template_no_nb.format(missing_pos='tail', triple=fwd_test_triple_str,
                                                                  known_pos='head',
                                                                  entity=hd_label, entity_intro=hd_intro,
                                                                  triple_facts=hd_triple_str,
                                                                  candidate_answers=candidate_tails_str)
                bwd_instruction = MC_prompt_template_no_nb.format(missing_pos='head', triple=bwd_test_triple_str,
                                                                  known_pos='tail',
                                                                  entity=tl_label, entity_intro=tl_intro,
                                                                  triple_facts=tl_triple_str,
                                                                  candidate_answers=candidate_heads_str)

        else:
            candidate_heads = []
            for ent in rerank_head_ids[:rerank_head_size]:
                candidate_heads.append(entity2wiki[ent]['label'])
            candidate_tails = []
            for ent in rerank_tail_ids[:rerank_tail_size]:
                candidate_tails.append(entity2wiki[ent]['label'])
            candidate_heads_str = ', '.join(candidate_heads)
            candidate_tails_str = ', '.join(candidate_tails)
            fwd_instruction = MC_prompt_template_no_ctx.format(missing_pos='tail', triple=fwd_test_triple_str,
                                                               known_pos='head',
                                                               entity=hd_label,
                                                               triple_facts=hd_triple_str,
                                                               candidate_answers=candidate_tails_str)
            bwd_instruction = MC_prompt_template_no_ctx.format(missing_pos='head', triple=bwd_test_triple_str,
                                                               known_pos='tail',
                                                               entity=tl_label,
                                                               triple_facts=tl_triple_str,
                                                               candidate_answers=candidate_heads_str)

        query = {}
        query['triple'] = stp
        query['seq_triple'] = [entity2wiki[head]['label'], rel2text[rel], entity2wiki[tail]['label']]
        query['mode'] = 'backward'
        query['gt'] = head
        query['instruction'] = bwd_instruction
        query['rerank_scope'] = rerank_head_size
        query['init_ranking'] = rerank_head_ids
        query['init_ranking_label'] = rerank_heads
        with open(bwd_jsonl_file_name, mode='a', encoding='utf-8') as f:
            json.dump(query, f)
            f.write('\n')

        query = {}
        query['triple'] = stp
        query['seq_triple'] = [entity2wiki[head]['label'], rel2text[rel], entity2wiki[tail]['label']]
        query['mode'] = 'forward'
        query['gt'] = tail
        query['instructions'] = fwd_instruction
        query['rerank_scope'] = rerank_tail_size
        query['init_ranking'] = rerank_tail_ids
        query['init_ranking_label'] = rerank_tails
        with open(fwd_jsonl_file_name, mode='a', encoding='utf-8') as f:
            json.dump(query, f)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate KGC Instructions')
    # parser.add_argument('--training_size', type=int, default=2,
    #                     help='number of triples used for training')
    parser.add_argument('--dataset', type=str, default='FB15k237')
    parser.add_argument('--reasoning', action='store_true', help='reasoning_part')
    parser.add_argument('--reasoning_ctx', action='store_true', help='reasoning_part')
    parser.add_argument('--use_ctx', action='store_true', help='reasoning_part')
    parser.add_argument('--kge', type=str, default='NBFnet')
    parser.add_argument('--llm', type=str, default='GPT')
    parser.add_argument('--num_rel_tps', type=int, default=5)
    parser.add_argument('--num_nb_tps', type=int, default=10)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=20466)
    parser.add_argument('--rerank_scope', type=int, default=10)

    args = parser.parse_args()
    main(args.dataset, args.kge, args.llm, args.start, args.end, args.rerank_scope, args.num_rel_tps,
         args.num_nb_tps, args.reasoning, args.reasoning_ctx, args.use_ctx)
