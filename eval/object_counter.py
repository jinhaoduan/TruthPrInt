import os
import json
import fire
import pandas as pd


def main(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    all_objects = 0
    all_objects_unique = 0
    hallucinated_objects = 0
    hallucinated_objects_unique = 0
    for i in data['sentences']:
        object_num = i["mscoco_generated_words"]
        object_num_unique = len(set(object_num))
        object_num = len(object_num)

        all_objects += object_num
        all_objects_unique += object_num_unique

        _hallucinated_objects = i['mscoco_hallucinated_words']
        hallucinated_objects += len(_hallucinated_objects)
        _unique_hallucinated_objects = []
        for _ho in _hallucinated_objects:
            _unique_hallucinated_objects.extend(list(set(_ho)))
        hallucinated_objects_unique += len(set(_unique_hallucinated_objects))

    df = pd.DataFrame({
        "all_objects": [all_objects],
        "all_objects_unique": [all_objects_unique],
        "hallucinated_objects": [hallucinated_objects],
        "hallucinated_objects_unique": [hallucinated_objects_unique]
    })
    df.to_csv(os.path.join(
        os.path.dirname(input_file), "object_counter.csv"
    ), index=False)


if __name__ == "__main__":
    fire.Fire(main)
