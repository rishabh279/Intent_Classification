import json
from constants import TRAINING_PATH, DATA_CORPUS_PATH, TESTING_PATH


def converting_data_rasa_format():
    '''Conversion of chatbot corpus into training data'''
    with open(DATA_CORPUS_PATH) as json_file:
        data = json.load(json_file)
        data['rasa_nlu_data'] = {'common_examples': data['sentences']}
        del data['lang']
        del data['name']
        del data['sentences']

        for json_data in data['rasa_nlu_data']['common_examples']:
            for entity_dict in json_data['entities']:
                entity_dict['start'] = json_data['text'].find(entity_dict['text'])
                entity_dict['stop'] = len(entity_dict['text']) + entity_dict['start']
                entity_dict['value'] = entity_dict['text']
                entity_dict['end'] = entity_dict['stop']
                del entity_dict['text']
                del entity_dict['stop']
    return data


def train_test_data(data):
    """Splitting into train test data in ratio 9:1"""
    count = 0
    test_data = {'rasa_nlu_data': {'common_examples': []}}
    for json_data in data['rasa_nlu_data']['common_examples']:
        if count < 20:
            if not json_data['training']:
                test_data['rasa_nlu_data']['common_examples'].append(json_data)
                data['rasa_nlu_data']['common_examples'].remove(json_data)
                count = count + 1

    return test_data


if __name__ == '__main__':
    data = converting_data_rasa_format()
    test_data = train_test_data(data)
    with open(TRAINING_PATH, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False)

    with open(TESTING_PATH, 'w') as outfile:
        json.dump(test_data, outfile, ensure_ascii=False)
