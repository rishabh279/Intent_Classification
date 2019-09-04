# Installation Instructions

1.Navigate to project folder.  

2.Run 'pip install requirements.txt'.  

3.Either run the run.py file or use the juypter notebook for testing the model.    
  
4.prediction() function gives the desired intent and entities of the given input.   

Model can further be improved with the following techniques:  

1.(https://github.com/topics/intent-classification?o=desc&s=stars)  

2.Since model is build and evaluated on the given dataset it can be further be improved by adding more data. These are some tools for data generation is rasa format:

(https://github.com/RasaHQ/rasa-nlu-trainer)  
(https://rodrigopivi.github.io/Chatito/)  
(https://yuukanoo.github.io/tracy/#/agents)  

3.In case we have less training samples or unlabeled data, we can improve it by Active Learning ie. predicting on the unseen data and then getting feedback from users.  

4.Since rasa uses word embedding for feature representation, the technique used for building word embedding is also important.  

5.Lastly we can build our own custom framework in order to get the best results for our custom domain.  

