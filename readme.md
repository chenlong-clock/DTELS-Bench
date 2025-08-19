#  DTELS: Towards Dynamic Granularity of TimeLine Summarization 
🔗 Preprint Link: https://arxiv.org/abs/2411.09297

🎉 This paper has been accepted as the NAACL 2025 main conference paper!

## �� Introduction
We extend the task of **Timeline Summarization (TLS)** to a new paradigm with timelines at dynamic granularities. We propose a benchmark containing **Dataset**, **Metrics** and **Evaluations**.

## 🏆 CCKS 2025 Shared Task - Simplified Evaluation Framework

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

## 📄 License

The CCKS 2025 demo evaluation framework is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. See the [LICENSE](LICENSE) file for details.

## ✨ Citation

```bibtex
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

## 🔧 Usage

### For CCKS 2025 Participants
Please refer to the `ccks2025_demo/` directory for the simplified evaluation tools and documentation.

### For Original DTELS Metrics
For researchers interested in implementing the complete DTELS evaluation framework as described in the paper, please follow the detailed methodology and mathematical formulations provided in the NAACL 2025 paper. The current demo provides a streamlined version for competition purposes.

