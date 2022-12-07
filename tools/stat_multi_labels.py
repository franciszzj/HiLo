import os
import sys
import json
import copy


frequency_dict = {
    'over': 39950,
    'in front of': 11433,
    'beside': 45859,
    'on': 54222,
    'in': 10051,
    'attached to': 20967,
    'hanging from': 3935,
    'on back of': 146,
    'falling off': 10,
    'going down': 118,
    'painted on': 192,
    'walking on': 7256,
    'running on': 1174,
    'crossing': 205,
    'standing on': 18754,
    'lying on': 1518,
    'sitting on': 5444,
    'flying over': 869,
    'jumping over': 81,
    'jumping from': 179,
    'wearing': 2982,
    'holding': 10466,
    'carrying': 2385,
    'looking at': 5351,
    'guiding': 90,
    'kissing': 17,
    'eating': 1283,
    'drinking': 117,
    'feeding': 70,
    'biting': 162,
    'catching': 135,
    'picking': 15,
    'playing with': 353,
    'chasing': 39,
    'climbing': 12,
    'cleaning': 22,
    'playing': 1934,
    'touching': 979,
    'pushing': 78,
    'pulling': 323,
    'opening': 8,
    'cooking': 19,
    'talking to': 428,
    'throwing': 183,
    'slicing': 139,
    'driving': 616,
    'riding': 2061,
    'parked on': 6795,
    'driving on': 5785,
    'about to hit': 572,
    'kicking': 75,
    'swinging': 743,
    'entering': 55,
    'exiting': 28,
    'enclosing': 600,
    'leaning on': 831,
}


def stat_multi_labels(in_file, out_file=None):
    if out_file is not None:
        fo = open(out_file, 'w')

    psg = json.load(open(in_file))
    data = psg['data']
    thing_classes = psg['thing_classes']
    stuff_classes = psg['stuff_classes']
    predicate_classes = psg['predicate_classes']
    object_classes = thing_classes + stuff_classes

    all_idx = 0
    repeat_idx = 0
    multi_labels_idx = 0
    spatial_rel_idx = 0
    double_spatial_rel_idx = 0
    change_idx = 0
    relation_num = [0 for _ in range(len(predicate_classes))]
    relation_list = []
    new_psg = copy.deepcopy(psg)
    for idx, d in enumerate(data):
        relations = d['relations']
        annotations = d['annotations']
        new_relations = copy.deepcopy(relations)
        for i, rel1 in enumerate(relations):
            all_idx += 1
            relation_list.append(rel1[2])
            relation_num[rel1[2]] += 1
            for j, rel2 in enumerate(relations):
                if i >= j:
                    continue
                if set(rel1) == set(rel2):
                    repeat_idx += 1
                    continue
                if (rel1[0] == rel2[0]) and (rel1[1] == rel2[1]):
                    multi_labels_idx += 1
                    sub_id = rel1[0]
                    obj_id = rel1[1]
                    sub = annotations[sub_id]
                    obj = annotations[obj_id]
                    subject = object_classes[sub['category_id']]
                    object = object_classes[obj['category_id']]
                    relation1 = predicate_classes[rel1[2]]
                    relation2 = predicate_classes[rel2[2]]
                    if rel1[2] < 6:
                        spatial_rel_idx += 1
                    if rel2[2] < 6:
                        spatial_rel_idx += 1
                    if (rel1[2] < 6) and (rel2[2] < 6):
                        double_spatial_rel_idx += 1
                    print('{} -> {} and {} have multi relations: {}, {}'.format(
                        d['file_name'], subject, object, relation1, relation2))

                    change_idx += 1
                    if frequency_dict[relation1] > frequency_dict[relation2]:
                        new_relations[i] = rel2
                        new_relations[j] = rel2
                    else:
                        new_relations[i] = rel1
                        new_relations[j] = rel1

                if all_idx % 100 == 0:
                    print('{} / {} / {} / {} / {} / {}'.format(multi_labels_idx,
                          repeat_idx, all_idx, spatial_rel_idx, double_spatial_rel_idx, change_idx))
        d['relations'] = new_relations
        new_psg['data'][idx] = copy.deepcopy(d)
    print('{} / {} / {} / {} / {} / {}'.format(multi_labels_idx, repeat_idx,
          all_idx, spatial_rel_idx, double_spatial_rel_idx, change_idx))
    for rel, rel_num in zip(predicate_classes, relation_num):
        print('{}: {}'.format(rel, rel_num))
        # print('{}: {}, {:.8f}'.format(rel, rel_num, rel_num / all_idx))
    if out_file is not None:
        json.dump(new_psg, fo)


if __name__ == '__main__':
    in_file = sys.argv[1]
    try:
        out_file = sys.argv[2]
    except:
        out_file = None
    stat_multi_labels(in_file, out_file)
