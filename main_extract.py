import argparse
import json
import logging
import os
import pickle
from tqdm import tqdm

# Import sklearn compatibility fix before any sklearn imports or pickle loads
import sklearn_compat

from utils.data import DTELSArticles, Timeline
from news_tls.clust import ClusterDateMentionCountRanker, ClusteringTimelineGenerator, TemporalMarkovClusterer
from news_tls.summarizers import CentroidOpt
from utils.tools import save_json
from news_tls.datewise import SupervisedDateRanker, PM_Mean_SentenceCollector, DatewiseTimelineGenerator  


def main(args):
    output_fn = f"{args.output_path}/{args.N}/{args.method}.jsonl"
    if not os.path.exists(os.path.dirname(output_fn)):
        os.makedirs(os.path.dirname(output_fn))
    logging.basicConfig(filename=f'{os.path.dirname(output_fn)}/log_{args.method}.log', level=logging.INFO)
    
    # Load articles
    articles_data = DTELSArticles("./articles")
    
    # Load gold reference timelines
    timelines = {}
    with open("./data/gold_reference.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            timeline = Timeline(data)
            timelines[timeline.id] = timeline
    
    # Create topics as list of (timeline, articles) pairs
    topics = []
    for tid in articles_data.keys():
        if tid in timelines:
            topics.append((timelines[tid], articles_data[tid]))
    
    exist_tls = []
    
    if os.path.exists(output_fn):
        with open(output_fn, "r") as f:
            exist_tls = [int(json.loads(line)['id']) for line in f.readlines()]

    if args.method == 'datewise':
        models_path = './news_tls/datewise/supervised_date_ranker.t17.pkl'
        with open(models_path, 'rb') as f:
            key_to_model = pickle.load(f)
        models = list(key_to_model.values())


        # load regression models for date ranking
        date_ranker = SupervisedDateRanker(method='regression')
        date_ranker.model = models[0]
        sent_collector = PM_Mean_SentenceCollector(
            clip_sents=5, pub_end=2)
        summarizer = CentroidOpt()
        tls_model = DatewiseTimelineGenerator(
            date_ranker=date_ranker,
            summarizer=summarizer,
            sent_collector=sent_collector,
            key_to_model = key_to_model
        )

    elif args.method == 'clust':
        cluster_ranker = ClusterDateMentionCountRanker()
        clusterer = TemporalMarkovClusterer()
        summarizer = CentroidOpt()
        tls_model = ClusteringTimelineGenerator(
            cluster_ranker=cluster_ranker,
            clusterer=clusterer,
            summarizer=summarizer,
            clip_sents=5,
            unique_dates=True,
        )
    else:
        raise ValueError(f'Method not found: {args.method}')


    n_topics = len(topics)
    for i, (tl, docs) in tqdm(enumerate(topics), total=n_topics):
        if tl.id in exist_tls:
            logging.info(f"timeline {tl.id} already exists, skipping.")
            continue
        logging.info(f"Processing {i+1}/{n_topics} {tl.id}")
        if args.N != 'N':
            max_dates = int(args.N)
        else:
            max_dates = len(tl)
        timeline = tls_model.predict(
            docs,
            max_dates=max_dates,
            max_summary_sents=1
            )
        timeline = Timeline(timeline, tl.id, tl.title)
        with open(output_fn, 'a', encoding='utf-8') as f:
            f.write(json.dumps(timeline.to_dict(), ensure_ascii=False) + '\n')
             

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['datewise', 'clust'])
    parser.add_argument('--output-path', default="./extract_output")
    parser.add_argument('--N', default='N')
    parser.add_argument("--log", default=True)
    main(parser.parse_args())
