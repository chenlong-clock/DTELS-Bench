# CCKS-DTELS Timeline Evaluation Scripts

This repository contains the official evaluation scripts for the CCKS-DTELS (Chinese Conference on Knowledge Graph and Semantic Computing - Dialogue-based Timeline Extraction and Summarization) shared task. These scripts are used to measure the quality of machine-generated timelines against a human-created reference.

## Evaluation Metrics

The evaluation is based on three core metrics:

1.  **Informativeness**: Measures how well the generated timeline covers the key information present in the reference timeline. It uses a Hungarian algorithm-based matching of events, considering both temporal proximity and content overlap (using ROUGE scores).

2.  **Granular Consistency**: Assesses the structural and temporal coherence of the generated timeline. It checks if the sequence and grouping of events are logical compared to the reference.

3.  **Factuality**: Verifies that the information presented in the generated timeline is factually supported by the provided source documents. This is done by checking for entailment between the timeline's atomic facts and the source text.

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### Running the Evaluation

The evaluation is triggered by the `py_entrance.sh` script. It requires an input JSON file to define the file paths and an output file to store the results.

```bash
# Usage: sh py_entrance.sh <input_params_file> <output_results_file>
sh py_entrance.sh input_param.json eval_result.json
```

-   `input_param.json`: Specifies the paths to the ground truth file (`standardFilePath`) and the submission file to be evaluated (`userFilePath`). See the existing file for an example.
-   `eval_result.json`: The file where the final evaluation scores will be written in JSON format.

## File Structure

-   `evaluate.py`: The main script that orchestrates the evaluation process.
-   `evaluate_timeline.py`: The core logic for calculating the informativeness, consistency, and factuality metrics.
-   `py_entrance.sh`: The main entry point shell script.
-   `requirements.txt`: A list of Python dependencies.
-   `answer.jsonl`: An example of a ground truth (standard) data file.
-   `result.jsonl`: An example of a submission file from a participating system.
-   `test_data/`: A directory containing more test examples.

## Input and Output Format

-   **Input (`.jsonl` files)**: The input files are in JSON Lines format, where each line is a JSON object representing a topic. Each topic has an `id` and a `timeline` composed of events.
-   **Output (`eval_result.json`)**: The script produces a single JSON object with the final scores, for example:

    ```json
    {
      "score": 0.85, 
      "scoreJson": {
        "score": 0.85,
        "info": 0.9,
        "granu": 0.8,
        "fact": 0.85
      },
      "success": true
    }
    ```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**. See the [LICENSE](LICENSE) file for details. This means it can be used and adapted for non-commercial research purposes, but any derivatives must be shared under the same license.
