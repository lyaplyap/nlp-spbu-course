import pandas as pd
import json
from os import path

DIRNAME = path.dirname(__file__)

PARQUET_DATA = path.join(DIRNAME, 'raw_data')
CONVERTED_DATA = path.join(DIRNAME, 'converted_data')
FINETUNING_DATA = path.join(DIRNAME, 'finetuning_data')

RESPONSE_MAX_LENGTH = 2000

def get_finetuning_data(df: pd.DataFrame):
    df = df.reset_index()
    data = []

    for _, row in df.iterrows():
        role = 'user'
        text = f'Напиши новость в стиле ИА "Панорама" для заголовка: {row['title']}'

        request = [{ 'role': role, 'text': text }]
        response = row['text']

        is_valid_response = len(response) <= RESPONSE_MAX_LENGTH

        if is_valid_response:
            data.append({ 'request': request, 'response': response })

    return data

def save_jsonl(list, file):
    with open(f'{file}.jsonl', 'w', encoding='utf-8') as f:
        json.dump(list[0], f, ensure_ascii=False)

        for i in range(1, len(list)):
            f.write('\n')
            json.dump(list[i], f, ensure_ascii=False)

if __name__ == '__main__':
    # Raw data
    df = pd.read_parquet(PARQUET_DATA)
    
    # Converted data
    df.to_json(CONVERTED_DATA, orient='records', force_ascii=False)

    # Data for fine-tuning
    data = get_finetuning_data(df)
    save_jsonl(data, FINETUNING_DATA)
