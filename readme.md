#  DTELS: Towards Dynamic Granularity of TimeLine Summarization 
🔗 Preprint Link: https://arxiv.org/abs/2411.09297

🎉 This paper has been accepted as the NAACL 2025 main conference paper! (https://aclanthology.org/2025.naacl-long.136/)

## 📌 Introduction
We extend the task of **Timeline Summarization (TLS)** to a new paradigm with timelines at dynamic granularities. We propose a benchmark containing **[Dataset](#Dataset)**, **[Metrics](#Metrics)** and **[Evaluations](#Evaluations)**.

## 📋 Table of Contents
- [🛠️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [✨ Citation](#-citation)

## ✨Citation 
```markdown
@misc{zhang2024dtelsdynamicgranularitytimeline,
      title={DTELS: Towards Dynamic Granularity of Timeline Summarization}, 
      author={Chenlong Zhang and Tong Zhou and Pengfei Cao and Zhuoran Jin and Yubo Chen and Kang Liu and Jun Zhao},
      year={2024},
      eprint={2411.09297},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.09297}, 
}
```

## 🛠️ Installation

### Prerequisites
- Python 3.10.18

### Setup Environment

1. **Clone the repository**
```bash
git clone https://github.com/chenlong-clock/DTELS-Bench.git
cd DTELS-Bench
```

2. **Create and activate conda environment**
```bash
conda create -n dtels-env python=3.10
conda activate dtels-env
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
import nltk
nltk.download('stopwords')
```

## 🚀 Usage

### Basic Timeline Generation

```python
from utils.data import DTELSArticles
from news_tls.datewise import DatewiseTimelineGenerator, MentionCountDateRanker, PM_Mean_SentenceCollector
from news_tls.summarizers import CentroidOpt

# Load articles data
articles = DTELSArticles(articles_path="articles")

# Initialize timeline generator
generator = DatewiseTimelineGenerator(
    date_ranker=MentionCountDateRanker(),
    sent_collector=PM_Mean_SentenceCollector(),
    summarizer=CentroidOpt()
)

# Generate timeline
timeline = generator.predict(
    collection=articles[1000],  # Use articles from timeline ID 1000
    max_dates=10,
    max_summary_sents=1
)

print("Generated Timeline:")
for item in timeline:
    print(f"Date: {item[1]}, Summary: {item[2]}")
```

### Running the Main Extraction Script

```bash
# Extract timelines using different methods
python main_extract.py \
    --method datewise \
    --N 10 \
    --output_path ./extract_output \
    --articles_path ./articles
```

### Command Line Arguments

- `--method`: Timeline generation method (`datewise`, `clustering`, etc.)
- `--N`: Maximum number of timeline nodes/dates to generate
- `--output_path`: Directory to save the generated timelines
- `--articles_path`: Path to the articles directory


## 📁 Project Structure

```
DTELS-Bench/
├── articles/                 # Article data files
├── news_tls/                # Core timeline summarization modules
│   ├── datewise.py          # Date-wise timeline generation
│   ├── clust.py             # Clustering-based methods
│   └── summarizers.py       # Text summarization utilities
├── utils/                   # Utility functions
│   ├── data.py              # Data loading and processing
│   └── tools.py             # Helper functions
├── time_nlp/               # Chinese time expression processing
├── main_extract.py         # Main extraction script
└── sklearn_compat.py       # scikit-learn compatibility fixes
```
