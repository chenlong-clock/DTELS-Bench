#  DTELS: Towards Dynamic Granularity of TimeLine Summarization 

## 📰 News

🔗 Preprint Link: https://arxiv.org/abs/2411.09297

🎉 This paper has been accepted as the NAACL 2025 main conference paper! (https://aclanthology.org/2025.naacl-long.136/)

🏆 We organized CCKS 2025 Shared Task - Event Timeline Generation for Social Media (https://tianchi.aliyun.com/competition/entrance/532361)

## 📌 Introduction
We extend the task of **Timeline Summarization (TLS)** to a new paradigm with timelines at dynamic granularities. We propose a benchmark containing **[Dataset](#Dataset)**, **[Metrics](#Metrics)** and **[Evaluations](#Evaluations)**.

## 📋 Table of Contents
- [🛠️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [🏆 CCKS 2025 Shared Task](#-ccks-2025-shared-task)
- [📁 Project Structure](#-project-structure)
- [🔧 Troubleshooting](#-troubleshooting)
- [📄 License](#-license)
- [✨ Citation](#-citation)

## 🛠️ Installation

### Prerequisites
- Python 3.10.18

### Setup Environment

1. **Clone the repository with submodules**
```bash
git clone --recursive https://github.com/chenlong-clock/DTELS-Bench.git
cd DTELS-Bench
```

**Or if you already cloned without submodules:**
```bash
git clone https://github.com/chenlong-clock/DTELS-Bench.git
cd DTELS-Bench
git submodule update --init --recursive
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

## 🏆 CCKS 2025 Shared Task

This repository includes a **simplified evaluation framework** for the **CCKS 2025 DTELS Shared Task**. The evaluation metrics have been streamlined and optimized for competition use while maintaining the core evaluation principles.

### ⚠️ Important Note on Evaluation Metrics

The `ccks2025_demo/` directory contains a **simplified version** of the DTELS evaluation metrics designed specifically for the CCKS 2025 shared task. This implementation focuses on:

- **Simplified Informativeness**: Using ROUGE-based matching with Hungarian algorithm
- **Streamlined Factuality**: Atomic proposition entailment checking 
- **Basic Granular Consistency**: Timeline structure evaluation

**For the complete and original DTELS evaluation metrics**, please refer to the methodology described in our NAACL 2025 paper. The full implementation follows the detailed guidelines and mathematical formulations presented in the paper.

### Quick Start for CCKS 2025 Demo

```bash
cd ccks2025_demo/
# Install dependencies (may need to install packages individually)
pip install rank-bm25 rouge-score scipy numpy tqdm

# Run evaluation
bash py_entrance.sh input_param.json eval_result.json
```

For detailed instructions, see the [CCKS 2025 Demo README](ccks2025_demo/README.md).

## 📁 Project Structure

```
DTELS-Bench/
├── articles/                 # Article data files  
├── ccks2025_demo/           # CCKS 2025 simplified evaluation framework
│   ├── evaluate.py          # Main evaluation orchestrator
│   ├── evaluate_timeline.py # Simplified core evaluation metrics
│   ├── py_entrance.sh       # Entry point script
│   ├── test_data/           # Sample test data
│   └── README.md            # Detailed demo documentation
├── news_tls/                # Core timeline summarization modules
├── utils/                   # Utility functions
├── main_extract.py         # Main extraction script
└── reference_timelines.jsonl # Reference timeline data
```

## 🔧 Troubleshooting

### time_nlp Submodule Issues

If you encounter issues with the `time_nlp` module not being visible or importable:

1. **Check if submodule is initialized:**
```bash
git submodule status
```

2. **Initialize/update submodules:**
```bash
git submodule update --init --recursive
```

3. **If submodule is empty or missing:**
```bash
git submodule sync
git submodule update --init --recursive
```

The `time_nlp` directory is a Git submodule pointing to the [Time_NLP project](https://github.com/zhanzecheng/Time_NLP) for Chinese time expression recognition.


## 📄 License

The CCKS 2025 demo evaluation framework is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. See the [LICENSE](LICENSE) file for details.

## ✨Citation 
```markdown
@inproceedings{zhang-etal-2025-dtels,
    title = "{DTELS}: Towards Dynamic Granularity of Timeline Summarization",
    author = "Zhang, Chenlong  and
      Zhou, Tong  and
      Cao, Pengfei  and
      Jin, Zhuoran  and
      Chen, Yubo  and
      Liu, Kang  and
      Zhao, Jun",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.136/",
    doi = "10.18653/v1/2025.naacl-long.136",
    pages = "2682--2703",
    ISBN = "979-8-89176-189-6"
}
```