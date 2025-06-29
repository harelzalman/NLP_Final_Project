import json, random
with open("fine_tuned\\phish_urls_train.json") as f:
    data = json.load(f)

print("records:", len(data))
print("few samples:", random.sample(data, 3))
