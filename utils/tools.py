from collections import Counter
from datetime import date
import os
import re
import string
import time
import json
import arrow
import numpy as np
from half_json.core import JSONFixer
from time_nlp import TimeNormalizer


def load_json(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip("\n")) for line in f]

def save_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if path.endswith(".json"):
        with open(path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    elif path.endswith(".jsonl"):
        with open(path, "w", encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

def save_jsonl(path, data):
    if path.endswith(".jsonl"):
        with open(path, "a", encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


def formatting_date(date, time_format = None):
    formated_time = None
    if time_format is None:
        time_formats = ["%Y-%m-%d %H:%M", "%Y年%m月%d日 %H:%M", "%Y-%m-%d", "%Y年%m月%d日", "%Y-%m", "%Y年%m月", "%Y", "%Y年"]
        for f in time_formats:
            try:
                formated_time = time.strptime(date, f)
            except:
                pass
    else:
        formated_time = time.strptime(date, time_format)
    if formated_time is None:
        print("date error:", date)
        raise ValueError
    return formated_time

def date_to_string(date, time_format = "%Y-%m-%d %H:%M"):
    if date.tm_hour == 0 or date.tm_min == 0:
        time_format = "%Y-%m-%d"
    return time.strftime(time_format, date)

def parse_json_format(input_str):
    while True:
        input_str = input_str.replace("\n ", "\n")
        if "\n " not in input_str:
            break
    input_str = input_str.replace("\n", "")
    if not input_str.endswith("}"):
        if not input_str.endswith("\""):
            input_str = input_str + "\""
    while input_str.count("{") > input_str.count("}"):
        input_str = input_str + "}"
    if not input_str.count("\"") % 2 == 0:
        input_str = input_str.replace("}}", "\"}}")
    return input_str


    
def split_chinese(para):
    para = re.sub('([。!?\?])([^”])',r"\1\n\2",para)
    para = re.sub('([。!?\?][”])([^,。!?\?])',r'\1\n\2',para)
    para = para.strip()
    return [p for p in para.split("\n") if p != ""]


def get_chinese_date(text):
    try:
        matched_date = arrow.get(text)
    except:
        matched_date = None
    if  matched_date is None:
        tn = TimeNormalizer()

        try:
            matched_time = json.loads(tn.parse(text))
        except:
            raise ValueError
        if 'error' in matched_time:
            raise ValueError
        if 'timestamp' in matched_time:
            matched_date = arrow.get(matched_time['timestamp'])
        elif 'timespan' in matched_time:
            matched_date = arrow.get(matched_time['timespan'][0])
        else:
            raise ValueError
    return matched_date