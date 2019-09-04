from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Interpreter
from constants import TRAINING_PATH, MODEL_PATH, TESTING_PATH
from rasa_nlu.test import run_evaluation
import json
import pandas as pd
import matplotlib.pyplot as plt


def hist_plot(input=pd.DataFrame()):
    """Plot number of examples per intent"""
    percent = pd.DataFrame.from_records(input['rasa_nlu_data']['common_examples'])['intent'].value_counts()
    percent.plot(kind='bar', figsize=(10, 8))
    plt.ylabel('Number of examples per intent')
    plt.xlabel('Intents')
    plt.title('Number of examples per intent')
    plt.grid()
    plt.show()


def train():
    """Training model on chatbot corpus"""
    training_data = load_data(TRAINING_PATH)
    trainer = Trainer(config.load('nlu_config.yml'))
    trainer.train(training_data)
    trainer.persist(path='models/', fixed_model_name='nlu_model')


def prediction(input_text=''):
    """Predict the intent and entities of the given sentence """
    interpreter = Interpreter.load(MODEL_PATH)
    result = interpreter.parse(input_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    hist_plot(pd.read_json(TRAINING_PATH))  # nb of examples per intent in the training data
    # train()
    run_evaluation(TESTING_PATH, MODEL_PATH, errors_filename='errors.json',
                   confmat_filename='confmat', intent_hist_filename='hist')
    prediction('when is the next train from winterstraße 12 to kieferngarten')
