import pandas as pd
import markovify
import time
from datasets import load_metric
import statistics

start_time = time.time() # start timer

'''import dataset with pandas'''
dataset = pd.read_csv('news-headlines.csv', encoding='utf-8')

'''load model'''
model = markovify.NewlineText(dataset.headline_text, state_size = 3)
model = model.compile()

'''generate predictions'''
predictions = []
for __ in range(10):
    predictions.append(model.make_sentence())
    [predictions.remove(i) for i in predictions if i == None]

[print(f'{i+1}. {prediction.title()}') for i, prediction in enumerate(predictions)]

'''BERT eval'''
metric = load_metric("bertscore")
results = metric.compute(predictions=predictions, references=dataset.headline_text[:len(predictions)], lang='en')

print(f'\nBERT metrics\n')
print(f"precision mean : {statistics.mean(results['precision'])}")
print(f"recall mean : {statistics.mean(results['recall'])}")
print(f"f1 mean : {statistics.mean(results['f1'])}")

print("\nTimer: %s seconds" % (time.time() - start_time)) # end timer