import json
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from transformers import CLIPTokenizer, CLIPTextModel
from prettytable import PrettyTable


class ObjectAwareRelationshipSimilarity(object):
    def __init__(self, model='bert-base-uncased'):
        self.object_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
            'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
            'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
            'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
            'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
            'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
            'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
            'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
            'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
            'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
            'food-other-merged', 'building-other-merged', 'rock-merged',
            'wall-other-merged', 'rug-merged'
        ]
        self.predicate_classes = [
            'over',
            'in front of',
            'beside',
            'on',
            'in',
            'attached to',
            'hanging from',
            'on back of',
            'falling off',
            'going down',
            'painted on',
            'walking on',
            'running on',
            'crossing',
            'standing on',
            'lying on',
            'sitting on',
            'flying over',
            'jumping over',
            'jumping from',
            'wearing',
            'holding',
            'carrying',
            'looking at',
            'guiding',
            'kissing',
            'eating',
            'drinking',
            'feeding',
            'biting',
            'catching',
            'picking',
            'playing with',
            'chasing',
            'climbing',
            'cleaning',
            'playing',
            'touching',
            'pushing',
            'pulling',
            'opening',
            'cooking',
            'talking to',
            'throwing',
            'slicing',
            'driving',
            'riding',
            'parked on',
            'driving on',
            'about to hit',
            'kicking',
            'swinging',
            'entering',
            'exiting',
            'enclosing',
            'leaning on',
        ]
        self._process_object_classes()

        # self.tokenizer = AutoTokenizer.from_pretrained(model)
        # self.model = AutoModel.from_pretrained(model)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.model = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32")

        self.cosine = nn.CosineSimilarity(dim=1)
        self.pairwise = nn.PairwiseDistance(p=2)

    def _process_object_classes(self):
        object_classes = []
        for oc in self.object_classes:
            oc = oc.replace('-stuff', '')
            oc = oc.replace('-other', '')
            oc = oc.replace('-merged', '')
            object_classes.append(oc)
        self.object_classes_changed = object_classes

    def get_embed(self, input):
        input_ids = self.tokenizer(input, return_tensors='pt')
        output = self.model(**input_ids)
        embed = output.pooler_output
        return embed

    def get_similarity(self, embed1, embed2):
        # similarity = self.cosine(embed1, embed2)
        similarity = self.pairwise(embed1, embed2)
        # similarity = similarity.sigmoid()
        # similarity = 1 / similarity
        return similarity.item()

    def get_two_objects_similarity(self, object1, object2):
        all_sentence = []
        for relation in self.predicate_classes:
            all_sentence.append('{} {} {}'.format(object1, relation, object2))
            # all_sentence.append(relation)

        similarity_matrix = torch.zeros(
            (len(self.predicate_classes), len(self.predicate_classes)))
        for i, relation1 in enumerate(self.predicate_classes):
            for j, relation2 in enumerate(self.predicate_classes):
                embed1 = self.get_embed(all_sentence[i])
                embed2 = self.get_embed(all_sentence[j])
                similarity = self.get_similarity(embed1, embed2)
                similarity_matrix[i, j] = similarity
                print('{}/{}, {}/{} - {} {} {}\tvs\t{} {} {} = {:.4f}'.format(
                    i, len(self.predicate_classes), j, len(
                        self.predicate_classes),
                    object1, relation1, object2,
                    object1, relation2, object2,
                    similarity))

        return similarity_matrix

    def get_all_objects_similarity(self):

        final_result = dict()

        for i, object1 in enumerate(self.object_classes_changed):
            for j, object2 in enumerate(self.object_classes_changed):
                similarity_matrix = self.get_two_objects_similarity(
                    object1, object2)
                similarity_matrix = similarity_matrix.numpy().tolist()
                print('{}/{}, {}/{} - object1: {}, object2: {}, similarity_matrix: \n{}'.format(
                    i, len(self.object_classes), j, len(self.object_classes),
                    object1, object2, similarity_matrix))
                key = '{}#{}'.format(
                    self.object_classes[i], self.object_classes[j])
                final_result[key] = similarity_matrix

        fo = open('./distance.json', 'w')
        json.dump(final_result, fo)


if __name__ == '__main__':
    oars = ObjectAwareRelationshipSimilarity(model='xlm-roberta-large')
    similarity_matrix = oars.get_two_objects_similarity('person', 'dog')
    print(similarity_matrix)
    # oars.get_all_objects_similarity()
