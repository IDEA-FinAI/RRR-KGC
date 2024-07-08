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

def combine_inter_and_rest_after_reason(inter_candidate_id_list, embedding_candidate_id_list, rest_embedding_candidate_id_list, cand_num):
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