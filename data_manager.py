import json

class DataManager:
    def __init__(self, dataset, embedding, forward):
        self.dataset = dataset
        self.embedding = embedding
        self.forward = forward
        self.base_path = f'./data/{dataset}'
        self.entity2detail = self.load_json('entity2detail.json')
        self.alignment = self.load_json('alignment.txt')
        self.e2idx, self.r2idx, self.idx2e, self.idx2r = self.load_entity_relation()
        self.train_set = self.load_relations('train.txt')
        
        self.train_lines = self.load_lines('train.txt')
        self.valid_lines = self.load_lines('valid.txt')
        self.test_lines = self.load_lines('test.txt')
        
        self.train_valid_lines = self.train_lines + self.valid_lines
        self.train_valid_set_tail_mapping = self.get_label_mapping(self.train_valid_lines)
        self.test_set_tail_mapping = self.get_id_mapping(self.test_lines)
        
        self.candidate_dict = self.load_candidates()

    def load_json(self, filename):
        with open(f'{self.base_path}/{filename}', 'r', encoding='utf-8') as file:
            return json.load(file)
    
    def load_lines(self, filename):
        with open(f'{self.base_path}/{filename}', 'r', encoding='utf-8') as file:
            return file.readlines()

    def load_relations(self, filename):
        relation_set = {}
        with open(f'{self.base_path}/{filename}', 'r', encoding='utf-8') as file:
            for line in file:
                head, relation, tail = line.strip().split('\t')
                if relation not in relation_set:
                    relation_set[relation] = []
                relation_set[relation].append((head, tail))
        return relation_set

    def load_candidates(self):
        retriever_file = f'{self.base_path}/{self.embedding}_retriever_candidate_tail.txt' if self.forward else f'{self.base_path}/{self.embedding}_retriever_candidate_head.txt'
        with open(retriever_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_entity_relation(self):
        with open(f'{self.base_path}/entities.txt', 'r', encoding='utf-8') as entity_file:
            entities = entity_file.read().splitlines()
            e2idx = {entity: idx for idx, entity in enumerate(entities)}
            idx2e = {idx: entity for idx, entity in enumerate(entities)}

        with open(f'{self.base_path}/relations.txt', 'r', encoding='utf-8') as relation_file:
            relations = relation_file.read().splitlines()
            r2idx = {relation: idx for idx, relation in enumerate(relations)}
            idx2r = {idx: relation for idx, relation in enumerate(relations)}

        return e2idx, r2idx, idx2e, idx2r

    def get_label_mapping(self, data_lines):
        label_mapping = {}
        for line in data_lines:
            head_id, relation_raw, tail_id = line.strip().split('\t')
            key = f"{head_id}\t{relation_raw}" if self.forward else f"{tail_id}\t{relation_raw}"
            if key not in label_mapping:
                label_mapping[key] = set()
            head_label = self.entity2detail[head_id]['label']
            tail_label = self.entity2detail[tail_id]['label']
            label_mapping[key].add(tail_label) if self.forward else label_mapping[key].add(head_label)
        return label_mapping
    
    def get_id_mapping(self, data_lines):
        id_mapping = {}
        for line in data_lines:
            head_id, relation_raw, tail_id = line.strip().split('\t')
            key = f"{head_id}\t{relation_raw}" if self.forward else f"{tail_id}\t{relation_raw}"
            if key not in id_mapping:
                id_mapping[key] = set()
            id_mapping[key].add(tail_id) if self.forward else id_mapping[key].add(head_id)
        return id_mapping

    def load_few_shot(self, cur_entity, relation, count):
        few_shot_pairs = []
        train_set = self.train_set

        if relation in train_set:
            for h, t in train_set[relation]:
                if (self.forward and h == cur_entity) or (not self.forward and t == cur_entity):
                    few_shot_pairs.append((h, t))
                    if len(few_shot_pairs) >= count:
                        return few_shot_pairs

            for h, t in train_set[relation]:
                if (h, t) not in few_shot_pairs:
                    few_shot_pairs.append((h, t))
                    if len(few_shot_pairs) >= count:
                        return few_shot_pairs

        return few_shot_pairs

    def load_sentence(self, head, relation_raw, relation, tail):
        dataset = self.dataset
        if dataset == 'FB15k237':
            last_relation = relation_raw.split('/')[-1]
            first_property = relation_raw.split('/')[2]
            if self.forward:
                return f'what is the {last_relation} of {first_property} {head}? The answer is '
            else:
                return f'{tail} is the {last_relation} of what {first_property}? The answer is '
        elif dataset == 'YAGO3-10':
            if self.forward:
                return f'{head} {relation} what? The answer is '
            else:
                return f'what {relation} {tail}? The answer is '