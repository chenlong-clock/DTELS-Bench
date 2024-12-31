from collections import OrderedDict
from time import struct_time
import arrow
from utils.tools import formatting_date, get_chinese_date, load_json, split_chinese
import os
class DTELSArticles(OrderedDict):
    def __init__(self, articles_ids=None, articles_path="articles", filter_low_quality_articles=True):
        for tf in os.listdir(articles_path):
            tid = int(tf.split(".")[0])
            if (tf.endswith(".jsonl") and articles_ids is not None and tid in articles_ids) or articles_ids is None:
                ti = int(tf.split(".")[0])
                self[ti] = Articles(load_json(os.path.join(articles_path, tf)), filter_low_quality_articles)

class Articles(list):
    def __init__(self, articles, filter_ids=False, lang='ch'):
        if filter_ids:
            if not os.path.exists("./filter_ids_dict.json"):
                raise FileNotFoundError("low quality article ids should be placed in \"./filter_ids_dict.json\" if filter_low_quality_articles == True")
            filter_ids = load_json("./filter_ids_dict.json")
        for a in articles:
            if isinstance(a, dict):
                a = Article(a, lang=lang)
            elif isinstance(a, Article):
                pass
            else:
                raise TypeError("Articles should be a list of dict or Article objects")
            if filter_ids and a.doc_id in filter_ids:
                continue
            self.append(a)
        self.sort_by_time()
        self.date_articles = OrderedDict()
        for a in self:
            if a.time not in self.date_articles:
                self.date_articles[a.time] = []
            self.date_articles[a.time].append(a)
        if len(self.date_articles) > 1:
            for date in self.date_articles:
                self.date_articles[date] = Articles(self.date_articles[date])
        self.start = self[0].time
        self.end = self[-1].time
        self.lang = lang

    def sort_by_time(self):
        self = sorted(self, key=lambda x: x.time)

    def find_near_articles(self, time, delta=7):
        cands_arts = []
        # find docs 1 week before and after the time
        if isinstance(time, arrow.Arrow):
            base_time = time
        else:
            base_time = arrow.get(time)
        for t in self.date_articles:
            try: 
                time_interval = abs(t - base_time).days
            except:
                continue
            if time_interval <= delta:
                cands_arts.extend(self.date_articles[t])
        return cands_arts
    
    def iter_by_date(self):
        for date, articles in self.date_articles.items():
            yield date, articles

    def __str__(self) -> str:
        if self.lang == "ch":
            prefix = "文章"
        elif self.lang == "en":
            prefix = "Article"
        for i, doc in enumerate(self):
            article_text = f"[{prefix} {i}]\n{doc}\n"                
            context += article_text
        return context
    

class Article:
    def __init__(self, doc_dict, lang="ch"):
        self.doc_id = doc_dict['doc_id']
        tl_id ,node_id, doc_type = self.doc_id.split("_")[:3]
        self.meta = {
            "timeline-id": tl_id,
            "node-id": node_id,
            "doc-type": doc_type,
            "lang": lang
        }
        self.title = doc_dict['title']
        self.time = arrow.get(*doc_dict['time'][:6])
        self.content = doc_dict['content']

    
    def __str__(self):
        time = self.time.format("YYYY-MM-DD")
        if self.meta.lang == "ch":
            return f"标题: {self.title}\n发布时间: {time}\n内容: {self.content}"
        elif self.meta.lang == "en":
            return f"Tittle: {self.title}\nRelease-time: {time}\nContent: {self.content}"
    
    def __getitem__(self, index):
        return self.content[index]

    def __len__(self):
        return len(self.content)
    
    def __slice__(self, start, end):
        return self.content[start:end]

class Timeline(object):
    def __init__(self, instance, tl_id=None, tl_title=None):
        self.meta = None
        self.timeline = []            

        if isinstance(instance, dict):
            self.title = instance['title']
            self.id = int(instance['id'])
            for item in instance['timeline']:
                try:
                    self.timeline.append(TimelineNode(item))
                except ValueError:
                    raise ValueError
            if "meta_timeline" in instance:
                try:
                    self.meta = {k: Timeline(v) for k, v in instance['meta_timeline'].items()}
                except:
                    self.meta = instance['meta_timeline']
        elif isinstance(instance, list):
            self.title = tl_title
            self.id = int(tl_id)
            for i, item in enumerate(instance):
                item = list(item)
                if item[0] == "":
                    item[0] = i
                try:
                    self.timeline.append(TimelineNode(item))
                except:
                    continue
        self.sort_timeline()
        self.times = sorted([item.time for item in self.timeline])
        if len(self.times) != 0:
            self.start_time = self.times[0]
            self.end_time = self.times[-1]
            assert self.start_time <= self.end_time
    

    def __getitem__(self, index):
        if isinstance(index, str) and index.isnumeric():
            index = int(index)
        return self.timeline[index]
    
    def __iter__(self):
        return iter(self.timeline)

    def __len__(self):
        return len(self.timeline)
    
    def __str__(self) -> str:
        # i. yyyy-mm-dd: summary
        node_str = ""
        for node in self.timeline:
            node_str += f"{node.id}." + node.__str__() + "\n"
        return node_str
    
    def sort_timeline(self):
        if self.timeline is not None:
            self.timeline = sorted(self.timeline, key=lambda x: x.time)
            for i, node in enumerate(self.timeline):
                self.timeline[i].id = i


    def to_dict(self):
        return_dict = {
            "title": self.title,
            "id": self.id,
            "timeline": [node._to_dict() for node in self.timeline]
        }
        if self.meta is None:
            return_dict["meta_timeline"] = None
        elif isinstance(self.meta, dict) and self.meta != {}:
            if '5' in self.meta:
                if isinstance(self.meta['5'], Timeline):
                    return_dict["meta_timeline"] = {k: v.to_dict() for k, v in self.meta.items()} 
                elif isinstance(self.meta['5'], dict):
                    return_dict['meta_timeline'] = self.meta
                else:
                    raise KeyError("no 5 in meta timeline")
        else:
           raise TypeError("meta_timeline should be a dict or a Timeline object")
        return return_dict
    
class TimelineNode:
    def __init__(self, node) -> None:
        self.atoms = None
        if isinstance(node, dict):
            self.id = int(node['id'])
            if isinstance(node['time'], str):
                self.time = get_chinese_date(node['time'])
            elif isinstance(node['time'], list):
                self.time = arrow.get(struct_time(node['time']))
            elif isinstance(node['time'], struct_time):
                self.time = arrow.get(node['time'])
            elif isinstance(node['time'], arrow.Arrow):
                self.time = node['time']
            if 'node_summary' in node:
                self.summary = node['node_summary']
            elif 'summary' in node:
                self.summary = node['summary']
            if "atomics" in node:
                self.atoms = node['atomics']
            elif "atoms" in node:
                self.atoms = node['atoms']
        elif isinstance(node, list):                
            self.id = int(node[0])
            self.time = arrow.get(formatting_date(node[1]))
            self.summary = node[2]
        
        if isinstance(self.summary, list):
            self.summary = "\n".join(self.summary)

    def _to_dict(self):
        return {
            "id": self.id,
            "time": str(self.time),
            "summary": self.summary,
            "atoms": self.atoms
        }

    def __eq__(self, __value: object) -> bool:
        return self.id == __value.id
    
    def __gt__(self, __value: object) -> bool:
        return self.time > __value.time
    
    def __str__(self):
        str_time = self.time.format("YYYY-MM-DD")
        return f"\"{str_time}\": \"{self.summary}\""
    