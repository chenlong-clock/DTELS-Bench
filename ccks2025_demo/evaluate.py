# coding=utf-8
import json
import sys
import numpy as np
# 错误字典，详细覆盖常见错误
error_msg = {
    1: "输入文件不存在或无法读取 (Input file not found or unreadable)",
    2: "输入文件格式错误 (Input file format error)",
    3: "输入文件内容不完整 (Input file incomplete)",
    4: "评测过程中发生未知错误 (Unknown error during evaluation)",
    5: "评测结果文件缺失或损坏 (Evaluation result file missing or corrupted)",
    6: "参数解析失败 (Failed to parfe parameters)",
    7: "评测脚本内部错误 (Internal evaluation script error)",
    8: "提交文件与测试集topic数量不符 (Submit file topics do not match test set)",
}

def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file, ensure_ascii=False)

def report_error_msg(detail, showMsg, out_p):
    error_dict = dict()
    error_dict['errorDetail'] = detail
    error_dict['errorMsg'] = showMsg  # 这个会透出给用户
    error_dict['score'] = 0
    error_dict['scoreJson'] = {}  # 出错时保持为空字典
    error_dict['success'] = False
    dump_2_json(error_dict, out_p)

def report_score(score, out_p, informativeness=None, granular_consistency=None, factuality=None):
    result = dict()
    result['success'] = True
    result['score'] = score

    # 这里{}里面的score注意保留，但可以增加其他key，比如这样：
    # result['scoreJson'] = {'score': score, 'aaaa': 0.1}
    score_json = {}
    if informativeness is not None:
        score_json['info'] = informativeness
    if granular_consistency is not None:
        score_json['granu'] = granular_consistency
    if factuality is not None:
        score_json['fact'] = factuality
    score_json['score'] = score
    result['scoreJson'] = score_json

    dump_2_json(result, out_p)

if __name__=="__main__":
    '''
      online evaluation
      
    '''
    in_param_path = sys.argv[1]
    out_path = sys.argv[2]

    # read submit and answer file from first parameter
    with open(in_param_path, 'r') as load_f:
        input_params = json.load(load_f)

    # 标准答案路径
    standard_path=input_params["fileData"]["standardFilePath"]
    print("Read standard from %s" % standard_path)

    # 选手提交的结果文件路径
    submit_path=input_params["fileData"]["userFilePath"]
    print("Read user submit file from %s" % submit_path)

    # 检查提交文件是否包含所有reference timeline的topic，且没有多余topic
    ref_topic_ids = set()
    try:
        with open(standard_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ref_topic_ids.add(obj.get('id'))
                except Exception as e:
                    print(f"Warning: Failed to parse line in standard file: {e}")
                    continue
        print(f"Successfully loaded {len(ref_topic_ids)} reference topics")
    except Exception as e:
        print(f"Error reading standard file: {e}")
        check_code = 1
        report_error_msg(f"Error reading standard file: {e}", error_msg.get(check_code, str(e)), out_path)
        sys.exit(1)
    submit_topic_ids = set()
    try:
        with open(submit_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    submit_topic_ids.add(obj.get('id'))
                except Exception as e:
                    print(f"Warning: Failed to parse line in submit file: {e}")
                    continue
        print(f"Successfully loaded {len(submit_topic_ids)} submit topics")
    except Exception as e:
        print(f"Error reading submit file: {e}")
        check_code = 1
        report_error_msg(f"Error reading submit file: {e}", error_msg.get(check_code, str(e)), out_path)
        sys.exit(1)
    missing_topics = ref_topic_ids - submit_topic_ids
    extra_topics = submit_topic_ids - ref_topic_ids
    if missing_topics:
        check_code = 8
        error_detail = f"提交文件缺少以下topic: {missing_topics}"
        report_error_msg(error_detail, error_msg.get(check_code, str(error_detail)), out_path)
        sys.exit(1)
    if extra_topics:
        check_code = 8
        error_detail = f"提交文件包含多余topic: {extra_topics}"
        report_error_msg(error_detail, error_msg.get(check_code, str(error_detail)), out_path)
        sys.exit(1)

    # 直接函数调用方式
    try:
        import asyncio
        import types
        import evaluate_timeline
        args = types.SimpleNamespace()
        args.test_file = submit_path
        args.reference_file = standard_path
        args.detail_output_file = './results/eval_detail.jsonl'
        args.output_file = './evaluation_results_simple.json'
        args.N = 'N'
        args.threshold = 0.5
        args.batch_size = 20
        args.force_reevaluate = True
        args.clear_detail_file = True
        args.fact_time_window = 60
        args.save_detail = False
        # 调用评测主函数，获取所有评测结果
        results, _ = asyncio.run(evaluate_timeline.evaluate_timelines(args))
        # 由evaluate.py统一写入评测结果文件
        # eval_output_path = './evaluation_results_simple.json'
        # with open(eval_output_path, 'w', encoding='utf-8') as f:
        #     json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        check_code = 7
        error_type = type(e).__name__
        error_detail = f"[{error_type}] {str(e)}"
        report_error_msg(error_detail, error_msg.get(check_code, str(e)), out_path)
        sys.exit(1)

    try:
        # 统计reference topics总数
        ref_topic_ids = set()
        with open(standard_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ref_topic_ids.add(obj.get('id'))
                except Exception:
                    continue
        total_topics = len(ref_topic_ids)
        # 统计评测结果中有分数的topic数量
        topic_keys = [k for k in results if isinstance(results[k], dict) and 'average' in results[k]]
        topic_scores = [results[k]['average'] for k in topic_keys]
        # 兼容overall
        if 'overall' in results and 'average' in results['overall']:
            score = results['overall']['average']
            informativeness = results['overall'].get('informativeness')
            granular_consistency = results['overall'].get('granular_consistency')
            factuality = results['overall'].get('factuality')
        else:
            # 按总topic数平均，缺失topic按0分
            if total_topics > 0:
                score = float(np.sum(topic_scores) / total_topics)
            else:
                score = 0.0
            informativeness = None
            granular_consistency = None
            factuality = None
        report_score(score, out_path, informativeness, granular_consistency, factuality)
    except Exception as e:
        check_code = 1
        error_type = type(e).__name__
        error_detail = f"[{error_type}] {str(e)}"
        report_error_msg(error_detail, error_msg.get(check_code, str(e)), out_path)
