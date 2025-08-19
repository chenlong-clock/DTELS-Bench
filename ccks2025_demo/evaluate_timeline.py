#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import asyncio
import argparse
from collections import Counter
import time
import logging
from rank_bm25 import BM25Okapi  # 添加BM25检索器
from tqdm import tqdm  # 进度条显示
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import asyncio
import argparse
from collections import Counter
from scipy.optimize import linear_sum_assignment
from scipy import optimize
import time
import logging
from rank_bm25 import BM25Okapi  # 添加BM25检索器
from tqdm import tqdm  # 进度条显示
from copy import deepcopy  # 添加deepcopy导入
from rouge_chinese import Rouge

import logging
import jieba
log_file = 'evaluation.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

async def compute_entailment_batch(text_pairs):
    """
    批量计算蕴含关系，使用ROUGE-L recall分数作为蕴含分数
    
    参数:
    - text_pairs: 要评估的文本对列表，每项为(text1, text2)
    
    返回:
    - 蕴含关系分数列表（ROUGE-L recall分数，范围0-1）
    """
    if not text_pairs:
        return []
    
    rouge = Rouge(metrics=['rouge-l'])
    results = []
    for text1, text2 in text_pairs:
        # 计算ROUGE-L recall分数
        tokens1 = ' '.join(jieba.cut(text1))
        tokens2 = ' '.join(jieba.cut(text2))
        score = rouge.get_scores(tokens1, tokens2)[0]['rouge-l']['r']
        results.append(score)
    return results

# 根据日期字符串计算序数值
def date_ordinal(date_str): # TODO: maybe have bugs when converting the data ordinal
    """将ISO格式的日期字符串转换为序数值"""
    if not date_str:
        return 0
    
    try:
        # 处理ISO格式日期字符串 (YYYY-MM-DDThh:mm:ss...)
        date_part = date_str.split("T")[0] if "T" in date_str else date_str
        year, month, day = map(int, date_part.split("-"))
        # 使用简单的日期序数计算 (年*366 + 月*31 + 日)
        return year * 366 + month * 31 + day
    except Exception as e:
        logger.warning(f"日期转换为序数值时出错: {e}, 日期: {date_str}")
        return 0


# 计算信息量指标
async def compute_informativeness(generated_timeline, reference_timeline):
    """
    计算信息量指标，使用时间权重和匈牙利算法优化，仅使用召回率计算
    
    参数:
    - generated_timeline: 生成的时间线节点列表
    - reference_timeline: 参考时间线节点列表
    
    返回:
    - informativeness: 信息量分数 (0-1之间的浮点数)
    - matches: 匹配的节点对列表
    - info_details: 详细的信息量评估结果
    """
    logger.info(f"开始计算信息量，生成节点数: {len(generated_timeline)}, 参考节点数: {len(reference_timeline)}")
    logger.info(f"评估流程：1) 计算时间距离成本矩阵；2) 计算原子事件召回率；3) 组合得到信息矩阵；4) 使用匈牙利算法寻找最优匹配；5) 计算总分")
    
    if not generated_timeline or not reference_timeline:
        return 0.0, [], {}
    
    # 详细记录每个节点的信息量评分
    info_details = {}
    recall_details = {}  # 记录详细的召回率信息
    
    # 1. 提取节点时间并转换为序数值
    gen_times = []
    for node in generated_timeline:
        time_str = node.get("time", "")
        gen_times.append(date_ordinal(time_str))
    gen_times = np.array(gen_times)
    
    ref_times = []
    for node in reference_timeline:
        time_str = node.get("time", "")
        ref_times.append(date_ordinal(time_str))
    ref_times = np.array(ref_times)
    
    # 2. 计算时间距离成本矩阵 (使用与原始代码相同的计算方式)
    # 使用 1/(|时间差|+1) 作为权重，时间越接近权重越大
    time_costs = np.zeros((len(generated_timeline), len(reference_timeline)))
    for i in range(len(generated_timeline)):
        for j in range(len(reference_timeline)):
            time_diff = abs(gen_times[i] - ref_times[j])
            time_costs[i, j] = 1 / (time_diff + 1)
    
    # 3. 计算内容匹配分数矩阵 (仅使用召回率)
    atomic_scores = np.zeros((len(generated_timeline), len(reference_timeline)))
    
    # 收集所有需要比较的文本对
    text_pairs_all = []
    node_indices = []  # 记录每个文本对属于哪个节点对
    
    # 按照Informative类的逻辑，使用完整的文本计算蕴含关系
    for i, gen_node in enumerate(generated_timeline):
        gen_id = gen_node.get("id", f"gen_{i}")
        gen_atoms = gen_node.get("atoms", [])
        
        if not gen_atoms:
            continue
        
        # 预测节点的原子事件拼接 (与Informative类一致)
        gen_text = "。".join([atom for atom in gen_atoms if atom.strip()])
        for j, ref_node in enumerate(reference_timeline):
            ref_id = ref_node.get("id", f"ref_{j}")
            ref_atoms = ref_node.get("atoms", [])
            
            if not ref_atoms:
                continue
            
            # 计算召回率 - 预测是否能够覆盖参考
            for ref_atom in ref_atoms:
                if ref_atom.strip():
                    # 每个参考原子事件都与生成节点的完整文本做比较
                    text_pairs_all.append((gen_text, ref_atom))
                    node_indices.append((i, j))
    
    # 批量计算蕴含分数
    if text_pairs_all:
        logger.info(f"计算召回率: 共有 {len(text_pairs_all)} 对文本需要比较")
        all_scores = await compute_entailment_batch(text_pairs_all)
        
        # 按节点对统计结果
        node_recall_stats = {}  # (i, j) -> [支持的原子事件数, 总原子事件数]
        
        # 初始化统计信息
        for i, gen_node in enumerate(generated_timeline):
            for j, ref_node in enumerate(reference_timeline):
                ref_atoms = ref_node.get("atoms", [])
                if ref_atoms:
                    valid_atoms = sum(1 for atom in ref_atoms if atom.strip())
                    if valid_atoms > 0:
                        node_recall_stats[(i, j)] = [0, valid_atoms]
        
        # 统计每个蕴含结果
        for idx, ((i, j), score) in enumerate(zip(node_indices, all_scores)):
            if (i, j) in node_recall_stats: 
                node_recall_stats[(i, j)][0] += score
        
        # 计算每对节点的召回率
        for (i, j), (supported, total) in node_recall_stats.items():
            recall = supported / total
            atomic_scores[i, j] = recall
            
            # 记录详细信息
            gen_id = generated_timeline[i].get("id", f"gen_{i}")
            ref_id = reference_timeline[j].get("id", f"ref_{j}")
            recall_details[f"{gen_id}_{ref_id}"] = {
                "score": recall,
                "atoms_recalled": supported,
                "total_atoms": total
            }
    # 4. 结合时间成本和原子分数得到信息矩阵 (与Informative类一致)
    info_matrix = atomic_scores * time_costs  # 注意转置以匹配维度
    
    # 5. 使用匈牙利算法找到最优匹配 (与Informative类一致)
    row_ind, col_ind = optimize.linear_sum_assignment(info_matrix, maximize=True)
    
    # 计算总体信息量分数 (与Informative类一致)
    informativeness = np.sum(info_matrix[row_ind, col_ind]) / len(generated_timeline) if generated_timeline else 0
    
    # 记录匹配结果
    matches = []
    for i, j in zip(row_ind, col_ind):
        gen_node = generated_timeline[i]
        ref_node = reference_timeline[j]
        match_score = info_matrix[i, j]
        
        if match_score > 0:
            matches.append((gen_node, ref_node, match_score))
            
            # 记录每个预测节点的最佳匹配
            gen_id = gen_node.get("id", f"gen_{i}")
            ref_id = ref_node.get("id", f"ref_{j}")
            
            info_details[gen_id] = {
                "best_match": ref_id,
                "score": match_score,
                "time_weight": time_costs[i, j],
                "recall": atomic_scores[i, j]
            }
    
    # 记录总体信息量分数和详细信息
    info_details["informativeness_score"] = informativeness
    info_details["recall_details"] = recall_details
    
    logger.info(f"计算完成，信息量分数: {informativeness:.4f}")
    
    return informativeness, matches, info_details

# 计算粒度一致性指标
def compute_granular_consistency(matches, reference_timeline):
    """
    计算粒度一致性指标，参照DGTLS-Bench中Granularity类的逻辑
    
    参数:
    - matches: 信息量评估函数返回的匹配列表，每项为(gen_node, ref_node, score)
    - reference_timeline: 参考时间线节点列表或Timeline对象
    
    返回:
    - granular_consistency: 粒度一致性得分 (0-1之间的浮点数)
    - granular_details: 详细的粒度评估结果
    # TODO: add 10, 5 to references
    """
    logger.info(f"开始计算粒度一致性，匹配数: {len(matches)}, 参考时间线节点数: {len(reference_timeline['timeline'])}")
    
    # 如果没有足够的匹配或参考时间线为空，返回默认值
    if len(matches) == 0 or not reference_timeline:
        logger.warning("没有匹配或参考时间线为空，返回默认粒度一致性分数")
        return 0.0, {"reason": "没有匹配或参考时间线为空"}
    
    # 对于生成节点数过少的情况直接返回0
    if len(matches) <= 1:
        logger.warning(f"生成节点数量过少 ({len(matches)}≤1)，粒度一致性无法计算，返回0")
        return 0.0, {"reason": "生成节点数量不足"}
    
    # 从匹配结果中提取生成节点和参考节点
    gen_nodes = [match[0] for match in matches]
    ref_nodes = [match[1] for match in matches]
    
    # 处理参考时间线，支持多粒度
    has_meta = False
    ref_timeline_nodes = reference_timeline
    
    # 检查是否有meta_timeline
    if hasattr(reference_timeline, 'meta') and reference_timeline.meta:
        has_meta = True
        # 深拷贝以避免修改原始数据
        temp_timeline = deepcopy(reference_timeline)
        ref_timeline_nodes = temp_timeline.timeline
    elif isinstance(reference_timeline, dict) and 'meta_timeline' in reference_timeline:
        has_meta = True
        temp_timeline = reference_timeline
        ref_timeline_nodes = reference_timeline.get('timeline', [])
    
    # 提取预测和参考时间线的时间信息并转换为序数值
    gen_times = []
    for node in gen_nodes:
        time_str = node.get("time", "")
        gen_times.append(date_ordinal(time_str))
    gen_times = np.array(gen_times)
    
    ref_times = []
    for node in ref_timeline_nodes:
        time_str = node.get("time", "")
        ref_times.append(date_ordinal(time_str))
    ref_times = np.array(ref_times)
    
    # 记录详细的粒度评估结果
    granular_details = {
        "node_mappings": {gen_nodes[i].get("id", f"gen_{i}"): ref_nodes[i].get("id", f"ref_{i}") 
                          for i in range(len(matches))},
        "matched_edges": [],
        "reference_edges": [],
        "edge_scores": {},
        "granu_score": {}
    }
    
    # 参考时间线的边界数量
    bound_cnt = len(ref_timeline_nodes) - 1
    granu_boundary = {"N": bound_cnt}  # N代表整个时间线
    granu_score = {"N": None}  # 初始化粒度得分
    
    # 处理多粒度评估 (如果存在meta_timeline)
    if has_meta:
        meta_data = {}
        if hasattr(reference_timeline, 'meta'):
            meta_data = reference_timeline.meta
        elif isinstance(reference_timeline, dict) and 'meta_timeline' in reference_timeline:
            meta_data = reference_timeline['meta_timeline']
        
        # 创建扩展的参考时间线和时间
        extended_nodes = ref_timeline_nodes.copy()
        extended_times = ref_times.copy()
        
        # 处理每个粒度级别
        for meta_key, meta_value in meta_data.items():
            if isinstance(meta_value, dict) and 'timeline' in meta_value:
                meta_tl = meta_value['timeline']
                granu_score[meta_key] = None
                
                # 添加新节点到扩展时间线
                for node in meta_tl:
                    node_id = node.get("id", "")
                    extended_nodes.append(node)
                    extended_times = np.append(extended_times, date_ordinal(node.get("time", "")))
                
                # 更新边界
                bound_cnt += len(meta_tl)
                granu_boundary[meta_key] = bound_cnt
        
        # 更新参考时间线和时间
        ref_timeline_nodes = extended_nodes
        ref_times = extended_times
    
    # 构建参考时间线的边集合
    ref_edges = set()
    for i in range(len(ref_timeline_nodes) - 1):
        current_edge = (ref_timeline_nodes[i].get("id", f"ref_{i}"), 
                        ref_timeline_nodes[i+1].get("id", f"ref_{i+1}"))
        ref_edges.add(current_edge)
        granular_details["reference_edges"].append({
            "from": ref_timeline_nodes[i].get("id", f"ref_{i}"),
            "to": ref_timeline_nodes[i+1].get("id", f"ref_{i+1}")
        })
    
    # 创建匹配得分矩阵
    eps = 1e-6  # 防止除零错误，与DGTLS-Bench一致
    info_matrix = np.zeros((len(gen_nodes), len(ref_timeline_nodes))) + eps
    
    # 填充匹配矩阵
    for i, match in enumerate(matches):
        gen_node = match[0]
        ref_node = match[1]
        score = match[2]
        
        # 在参考时间线中找到匹配节点的索引
        for j, node in enumerate(ref_timeline_nodes):
            if node.get("id", "") == ref_node.get("id", ""):
                info_matrix[i, j] = score + eps  # 添加小值防止除零
                break
    
    # 计算时间成本矩阵 - 使用相同的方法
    time_costs = np.zeros((len(gen_nodes), len(ref_timeline_nodes)))
    for i in range(len(gen_nodes)):
        for j in range(len(ref_timeline_nodes)):
            time_diff = abs(gen_times[i] - ref_times[j])
            time_costs[i, j] = 1 / (time_diff + 1)  # 时间越近，权重越大
    
    # 结合匹配得分和时间成本
    info_total = info_matrix * time_costs
    
    # 按照DGTLS-Bench中的逻辑构建边缘信息
    # 内部边缘 - 使用切片操作而非循环
    inner_edge = np.zeros((len(gen_nodes), len(ref_timeline_nodes)))
    if len(gen_nodes) > 1 and len(ref_timeline_nodes) > 1:
        inner_edge[1:, 1:] = info_total[1:, 1:]
    
    # 边缘信息矩阵 - 原始节点加上后继节点的影响
    edge_info = np.zeros((len(gen_nodes), len(ref_timeline_nodes)))
    if len(gen_nodes) > 1 and len(ref_timeline_nodes) > 1:
        edge_info[:-1, :-1] = info_total[:-1, :-1] + inner_edge[1:, 1:]
    
    # 处理边界 - 与DGTLS-Bench一致
    for key in granu_boundary:
        if granu_boundary[key] < edge_info.shape[1]:
            edge_info[:, granu_boundary[key]] = 0
    
    # 使用匈牙利算法找到最优匹配 - 注意这里使用了最大化
    row_ind, col_ind = optimize.linear_sum_assignment(edge_info, maximize=True)
    mounted_indices = col_ind  # 使用匹配到的列索引
    
    # 计算粒度得分 - 与DGTLS-Bench完全一致
    matched_count = 0
    for meta in granu_boundary:
        # 如果边界索引在匹配索引中，移除它
        if granu_boundary[meta] in mounted_indices:
            mounted_indices = mounted_indices[mounted_indices != int(granu_boundary[meta])]
        
        # 计算在边界前的匹配节点数
        counts = np.sum(mounted_indices < granu_boundary[meta])
        
        # 修复：确保分子不会超过分母，分母最小为1，得分不会大于1
        new_matches = counts - matched_count
        max_possible = max(1, len(gen_nodes) - 1)
        granu_score[meta] = min(new_matches, max_possible) / max_possible
        matched_count = counts
    
    # 确保边匹配的严谨性，检查真正匹配的边
    gen_edges = []
    for i in range(len(gen_nodes) - 1):
        gen_edges.append((gen_nodes[i].get("id", f"gen_{i}"), gen_nodes[i+1].get("id", f"gen_{i+1}")))
    
    logger.info(f"生成边数: {len(gen_edges)}, 参考边数: {len(ref_edges)}")
    
    # 记录匹配的边
    matched_edges = set()
    total_edge_score = 0.0
    
    for i, j in zip(row_ind, col_ind):
        # 只处理有效的索引
        if i < len(gen_nodes) - 1 and j < len(ref_timeline_nodes) - 1:
            gen_id = gen_nodes[i].get("id", f"gen_{i}")
            gen_next_id = gen_nodes[i+1].get("id", f"gen_{i+1}")
            gen_edge = (gen_id, gen_next_id)
            
            ref_id1 = ref_timeline_nodes[j].get("id", f"ref_{j}")
            ref_id2 = ref_timeline_nodes[j+1].get("id", f"ref_{j+1}")
            ref_edge = (ref_id1, ref_id2)
            
            # 检查是否匹配到了参考边，并确保匹配质量
            if ref_edge in ref_edges:
                edge_score = float(edge_info[i, j])
                
                # 只有当边匹配分数超过阈值时才计入
                MIN_EDGE_SCORE = eps * 10  # 使用eps的10倍作为最小阈值
                if edge_score > MIN_EDGE_SCORE:
                    matched_edges.add(ref_edge)
                    total_edge_score += edge_score
                    
                    granular_details["matched_edges"].append({
                        "gen_from": gen_id,
                        "gen_to": gen_next_id,
                        "ref_from": ref_id1,
                        "ref_to": ref_id2,
                        "edge_score": edge_score
                    })
                    
                    # 记录边得分
                    edge_key = f"{ref_id1}_{ref_id2}"
                    granular_details["edge_scores"][edge_key] = edge_score
    
    # 计算总体粒度一致性 - 使用N粒度的得分
    granular_consistency = granu_score.get("N", 0.0)
    
    # 添加详细信息
    granular_details["granular_consistency_score"] = granular_consistency
    granular_details["matched_edge_count"] = len(matched_edges)
    granular_details["total_edge_count"] = len(ref_edges)
    granular_details["matched_edge_ratio"] = len(matched_edges) / len(ref_edges) if ref_edges else 0
    granular_details["average_edge_score"] = total_edge_score / len(matched_edges) if matched_edges else 0
    granular_details["granu_score"] = granu_score
    
    logger.info(f"粒度一致性计算完成，得分: {granular_consistency:.4f}")
    logger.info(f"匹配边数: {len(matched_edges)}/{len(ref_edges)}, 匹配率: {granular_details['matched_edge_ratio']:.4f}")
    
    return granular_consistency, granular_details

# 用于事实性评估的辅助类
class FactChecker:
    """
    用于事实性评估的检查器类，提供更清晰的状态管理
    """
    def __init__(self, threshold=0.3, time_window=60):
        self.threshold = threshold
        self.time_window = time_window  # 新增：时间窗口（天）
        self.retriever = None
    
    async def check_atomic_facts(self, doc_text, atoms, states):
        """
        检查文档文本是否支持原子事件
        
        参数:
        - doc_text: 文档文本
        - atoms: 原子事件列表
        - states: 当前原子事件的状态字典 {索引: 状态(0或1)}
        
        返回:
        - 更新后的状态字典
        """
        # 如果所有原子事件都已经被判定为事实，直接返回
        if sum(states.values()) == len(atoms):
            return states
            
        # 为每个未被判定为事实的原子事件创建文本对
        text_pairs = []
        indices = []
        for i, atom in enumerate(atoms):
            if states[i] == 0 and atom.strip():  # 只评估尚未被判定为事实的原子事件
                text_pairs.append((doc_text, atom))
                indices.append(i)
        
        if not text_pairs:
            return states
            
        # 批量计算蕴含分数
        entailment_scores = await compute_entailment_batch(text_pairs)
        
        # 更新状态
        for idx, score in zip(indices, entailment_scores):
            if score > self.threshold:
                states[idx] = 1  # 将该原子事件标记为事实
        
        return states
    
    def retrieve_relevant_docs(self, docs, node_time, node_summary, atoms, max_docs=5):
        """
        检索与节点相关的文档
        
        参数:
        - docs: 文档列表
        - node_time: 节点时间
        - node_summary: 节点摘要
        - atoms: 原子事件列表
        - max_docs: 最大文档数量
        
        返回:
        - 检索到的相关文档段落列表
        """
        # 1. 按时间筛选文档
        time_filtered_docs = self._filter_docs_by_time(docs, node_time)
        
        # 2. 将文档分割成段落
        doc_segments = self._segment_documents(time_filtered_docs)
        
        # 3. 使用BM25检索与节点相关的段落
        retrieved_segments = self._retrieve_segments(doc_segments, node_summary, atoms, max_segments=3)
        
        return retrieved_segments
    
    def _filter_docs_by_time(self, docs, node_time):
        """按时间筛选文档，基于天数差异"""
        if not node_time:
            return docs
        try:
            node_date = node_time.split("T")[0]
            node_year, node_month, node_day = map(int, node_date.split("-"))
            from datetime import date
            node_dt = date(node_year, node_month, node_day)
            time_filtered = []
            for doc in docs:
                doc_time = doc.get("time", "")
                if not doc_time:
                    continue
                doc_date = doc_time.split("T")[0]
                if not doc_date:
                    continue
                try:
                    doc_year, doc_month, doc_day = map(int, doc_date.split("-"))
                    doc_dt = date(doc_year, doc_month, doc_day)
                    day_diff = abs((doc_dt - node_dt).days)
                    if day_diff <= self.time_window:
                        time_filtered.append(doc)
                except Exception:
                    pass
            return time_filtered if time_filtered else docs
        except Exception:
            return docs
    
    def _segment_documents(self, docs, segment_size=256):
        """将文档分割成适当大小的段落"""
        segments = []
        for doc in docs:
            content = doc.get("content", "")
            title = doc.get("title", "")
            full_text = title + "\n" + content
            
            # 分割文档
            for i in range(0, len(full_text), segment_size):
                if i + segment_size > len(full_text):
                    segment = full_text[i:]
                else:
                    segment = full_text[i:i+segment_size]
                
                if len(segment.strip()) > 20:  # 忽略太短的段落
                    segments.append(segment)
        
        return segments
    
    def _retrieve_segments(self, segments, node_summary, atoms, max_segments=3):
        """使用BM25检索与节点摘要最相关的段落"""
        if not segments:
            return []
        try:
            # 构建查询文本
            query = node_summary if node_summary else "。".join([atom for atom in atoms if atom.strip()])
            # 创建BM25检索器
            tokenized_segments = [list(segment) if isinstance(segment, list) else segment.split() for segment in segments]
            self.retriever = BM25Okapi(tokenized_segments)
            # 检索相关段落
            tokenized_query = list(query) if isinstance(query, list) else query.split()
            top_n = min(max_segments, len(segments))
            top_segments = self.retriever.get_top_n(tokenized_query, segments, top_n)
            return top_segments
        except Exception as e:
            logger.warning(f"BM25检索出错: {e}，返回前几个段落")
            return segments[:min(max_segments, len(segments))]

# 计算事实性指标，使用API
async def compute_factuality(generated_timeline, reference_data, topic_id, threshold=0.5, time_window=60):
    logger.info("\n===== 开始事实性评估 =====")
    logger.info(f"评估标准: 蕴含阈值设为 {threshold} (值越高标准越严格)，时间窗口: {time_window} 天")
    fact_checker = FactChecker(threshold, time_window)
    fact_details = {}
    topics = {}
    for node in generated_timeline:
        t_id = node.get("topic_id", topic_id)
        if t_id not in topics:
            topics[t_id] = []
        topics[t_id].append(node)
    total_nodes = sum(len(nodes) for nodes in topics.values())
    processed_nodes = 0
    for t_id, nodes in topics.items():
        logger.info(f"处理主题 {t_id}, 节点数量: {len(nodes)}")
        processed_nodes += len(nodes)
        docs = reference_data[t_id]["docs"] if "docs" in reference_data[t_id] else None
        if not docs:
            logger.warning(f"主题 {t_id} 没有文档，跳过事实性评估")
            continue
        node_tasks = [evaluate_node_factuality(node, docs, fact_checker, processed_nodes, total_nodes, logger) for node in nodes]
        node_results = await asyncio.gather(*node_tasks)
        for node_id, states in node_results:
            if states is not None:
                fact_details[node_id] = states
    if len(fact_details) == 0:
        factuality = 0
    else:
        node_scores = [fact_details[node_id]['score'] for node_id in fact_details if node_id != 'fact_score']
        factuality = sum(node_scores) / len(node_scores) if node_scores else 0
    fact_details['fact_score'] = factuality
    logger.info(f"整体事实性得分: {factuality:.4f}")
    return factuality, fact_details

# 并行节点事实性评估函数
evaluate_node_factuality = None  # 占位，防止未定义错误

async def evaluate_node_factuality(node, docs, fact_checker, processed_nodes, total_nodes, logger):
    node_id = node.get("id", "unknown")
    node_time = node.get("time", "")
    node_summary = node.get("summary", "")
    gen_atoms = node.get("atoms", [])
    if not gen_atoms:
        logger.debug(f"节点 {node_id} 没有原子事件，跳过")
        return node_id, None
    logger.info(f"评估节点 {node_id} ({processed_nodes}/{total_nodes})")
    retrieved_segments = fact_checker.retrieve_relevant_docs(
        docs, node_time, node_summary, gen_atoms
    )
    if not retrieved_segments:
        logger.warning(f"节点 {node_id} 没有找到相关文档段落，跳过")
        return node_id, None
    states = {i: 0 for i in range(len(gen_atoms))}
    for segment in retrieved_segments:
        states = await fact_checker.check_atomic_facts(segment, gen_atoms, states)
        if sum(states.values()) == len(gen_atoms):
            break
    states['score'] = sum(states.values()) / len(gen_atoms)
    logger.info(f"节点 {node_id} 事实性得分: {states['score']:.4f} ({sum(states.values())}/{len(gen_atoms)})")
    return node_id, states

# 主函数：评估生成的时间线
async def evaluate_timelines(args):
    try:
        # 初始化API处理器
        
        # 准备节点详情输出文件
        detail_file = args.detail_output_file
        try:
            # 确保目录存在
            detail_dir = os.path.dirname(detail_file)
            if detail_dir and args.save_detail:
                os.makedirs(detail_dir, exist_ok=True)
                logger.info(f"已确保节点详细信息输出目录存在: {detail_dir}")
            
            # 如果指定了清空节点详情文件，则执行清空操作
            if args.clear_detail_file and os.path.exists(detail_file):
                # 清空文件内容
                with open(detail_file, 'w', encoding='utf-8') as f:
                    f.write("")
                logger.info(f"已清空节点详细信息文件: {detail_file}")
            elif not os.path.exists(detail_file):
                # 如果文件不存在，创建一个空文件
                with open(detail_file, 'w', encoding='utf-8') as f:
                    pass
                logger.info(f"已创建节点详细信息文件: {detail_file}")
            else:
                logger.info(f"将使用现有节点详细信息文件: {detail_file}")
        except Exception as e:
            logger.error(f"准备节点详细信息文件时出错: {e}")
            # 降级到当前目录
            detail_file = f"./node_details_simple.jsonl"
            logger.warning(f"将使用备用的节点详细信息文件: {detail_file}")
        
        logger.info(f"节点详细信息将保存至: {detail_file}")
        
        # 加载参考时间线
        reference_data = {}
        with open(args.reference_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                topic_id = str(data.get("id", ""))
                if topic_id and "timeline" in data and "meta_timeline" in data:
                    reference_data[topic_id] = {
                        "timeline": data["timeline"],
                        "meta_timeline": data["meta_timeline"],
                        "docs": data['docs']
                    }
        
        logger.info(f"已加载 {len(reference_data)} 个参考时间线")
        logger.info(f"并行批处理大小: {args.batch_size}")
        
        # 加载测试数据
        test_data = []
        with open(args.test_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                test_data.append(data)
        
        logger.info(f"已加载 {len(test_data)} 个测试样本")
        
        # 准备输出文件
        output_file = args.output_file
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 如果文件已存在，读取已有内容
        existing_results = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                    logger.info(f"已加载现有评估结果，包含 {len(existing_results)} 个主题")
            except Exception as e:
                logger.warning(f"读取现有结果文件时出错: {e}，将创建新文件")
        
        # 评估结果
        results = existing_results.copy()
        
        # 使用tqdm显示总体进度
        overall_pbar = tqdm(test_data, desc="评估话题进度", unit="话题")
        
        # 对每个测试样本进行评估
        for sample in overall_pbar:
            try:
                topic_id = str(sample.get("id", ""))
                overall_pbar.set_description(f"评估话题 {topic_id}")
                if topic_id in results and not args.force_reevaluate:
                    logger.info(f"主题 {topic_id} 已有评估结果，跳过评估")
                    continue
                if topic_id not in reference_data:
                    logger.warning(f"话题ID {topic_id} 在参考数据中不存在，跳过评估")
                    continue
                logger.info(f"\n============= 开始评估话题 {topic_id} =============")
                generated_timeline = sample.get("timeline", [])
                reference = reference_data[topic_id]
                original_timeline = reference["timeline"]
                if args.N == "N":
                    reference_timeline = original_timeline
                    logger.info(f"使用原始参考时间线 (节点数: {len(original_timeline)})")
                else:
                    if reference["meta_timeline"] and args.N in reference["meta_timeline"]:
                        reference_timeline = reference["meta_timeline"][args.N]["timeline"]
                        logger.info(f"使用粒度 {args.N} 的meta_timeline (节点数: {len(reference_timeline)})")
                    else:
                        reference_timeline = original_timeline
                        logger.warning(f"在参考时间线中未找到粒度为 {args.N} 的meta_timeline，使用原始时间线 (节点数: {len(reference_timeline)})")
                logger.info(f"生成节点数: {len(generated_timeline)}, 参考节点数: {len(reference_timeline)}, 粒度设置: {args.N}")
                overall_pbar.set_postfix(阶段="计算信息量")
                informativeness, matches, info_details = await compute_informativeness(generated_timeline, reference_timeline)
                overall_pbar.set_postfix(阶段="计算粒度一致性")
                granular_consistency, granular_details = compute_granular_consistency(matches, reference)
                overall_pbar.set_postfix(阶段="计算事实性")
                for node in generated_timeline:
                    if "topic_id" not in node:
                        node["topic_id"] = topic_id
                    # 传递 reference 作为 reference_timeline，供 compute_factuality 使用
                    node["reference_timeline"] = reference
                factuality, fact_details = await compute_factuality(
                    generated_timeline, reference_data, topic_id, threshold=args.threshold, time_window=args.fact_time_window
                )
                results[topic_id] = {
                    "informativeness": informativeness,
                    "info_details": info_details,
                    "granular_consistency": granular_consistency,
                    "granular_details": granular_details,
                    "factuality": factuality,
                    "fact_details": fact_details,
                    "average": (informativeness + granular_consistency + factuality) / 3
                }
                logger.info(f"\n话题ID {topic_id} 评估完成:")
                logger.info(f"信息量={informativeness:.4f}")
                logger.info(f"粒度一致性={granular_consistency:.4f}")
                logger.info(f"事实性={factuality:.4f}")
                logger.info(f"平均分={results[topic_id]['average']:.4f}")
                overall_pbar.set_postfix(平均得分=f"{results[topic_id]['average']:.4f}")
                if hasattr(args, 'save_detail'):
                    if args.save_detail:
                        save_node_result(
                            topic_id=topic_id,
                            info_score=informativeness,
                            granularity_score=granular_consistency,
                            factuality_score=factuality,
                            matches=matches,
                            fact_details=fact_details,
                            file_path=detail_file
                        )
                else:
                    save_node_result(
                        topic_id=topic_id,
                        info_score=informativeness,
                        granularity_score=granular_consistency,
                        factuality_score=factuality,
                        matches=matches,
                        fact_details=fact_details,
                        file_path=detail_file
                    )
            except Exception as e:
                logger.error(f"评估话题 {sample.get('id', '')} 时出错: [{type(e).__name__}] {e}")
                continue
        # 计算总体得分
        overall_results = None
        if results:
            topic_results = {k: v for k, v in results.items() if k != "overall"}
            if topic_results:
                overall_results = {
                    "informativeness": np.mean([result["informativeness"] for result in topic_results.values()]),
                    "granular_consistency": np.mean([result["granular_consistency"] for result in topic_results.values()]),
                    "factuality": np.mean([result["factuality"] for result in topic_results.values()]),
                    "average": np.mean([result["average"] for result in topic_results.values()])
                }
                results["overall"] = overall_results
        else:
            logger.warning("没有找到可评估的样本")
        logger.info(f"评估完成，总体得分: {overall_results['average'] if overall_results else '无'}")
        return results, overall_results
    except Exception as e:
        logger.error(f"评测主流程发生异常: [{type(e).__name__}] {e}")
        raise

# 添加保存单个主题节点结果的函数
def save_node_result(topic_id, info_score, granularity_score, factuality_score, 
                     matches=None, fact_details=None, file_path="./results/g_info.jsonl"):
    """
    将单个主题的评估结果保存到JSONL文件
    
    参数:
    - topic_id: 主题ID
    - info_score: 信息量得分
    - granularity_score: 粒度一致性得分
    - factuality_score: 事实性得分
    - matches: 匹配的节点对列表
    - fact_details: 事实性详细信息
    - file_path: 输出文件路径
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"已创建目录: {directory}")
        
        # 检查文件是否存在
        file_exists = os.path.exists(file_path)
        
        # 构建记录
        record = {
            "topic_id": topic_id,
            "informativeness": info_score,
            "granularity": granularity_score,
            "factuality": factuality_score,
            "average": (info_score + granularity_score + factuality_score) / 3,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        
        # 添加节点详细信息
        if fact_details:
            # 过滤出节点级别的事实性详情
            node_facts = {k: v for k, v in fact_details.items() 
                          if k != "fact_score" and isinstance(v, dict)}
            if node_facts:
                record["node_factuality"] = node_facts
        
        # 添加匹配信息
        if matches:
            matched_nodes = []
            for gen_node, ref_node, score in matches:
                matched_nodes.append({
                    "gen_id": gen_node.get("id", ""),
                    "ref_id": ref_node.get("id", ""),
                    "score": score
                })
            record["matched_nodes"] = matched_nodes
        
        # 使用追加模式写入JSONL文件
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        if not file_exists:
            logger.info(f"已创建并写入节点详细信息文件: {file_path}")
        else:
            logger.info(f"已将主题 {topic_id} 的节点详细信息追加至 {file_path}")
    
    except Exception as e:
        logger.error(f"保存节点详细信息时出错: {e}")
        # 尝试使用备用路径
        backup_path = f"./node_details_{topic_id}.json"
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            logger.warning(f"已将节点详细信息保存到备用文件: {backup_path}")
        except Exception as backup_error:
            logger.error(f"保存到备用文件也失败: {backup_error}")
            # 最后尝试直接打印到日志
            logger.info(f"主题 {topic_id} 的评估结果: {json.dumps(record, ensure_ascii=False)}")

def parse_args():
    parser = argparse.ArgumentParser(description='评估生成的时间线')
    
    # 文件路径参数
    parser.add_argument('--test_file', type=str, default='./test.jsonl',
                        help='测试数据文件路径')
    parser.add_argument('--reference_file', type=str, default='./data/dtels/test_reference_timelines_with_docs.jsonl',
                        help='参考时间线文件路径')
    parser.add_argument('--output_file', type=str, default='./evaluation_results_simple.json',
                        help='评估结果输出文件路径')
    parser.add_argument('--detail_output_file', type=str, default='./results/g_info.jsonl',
                        help='节点详细信息输出文件路径，包含节点级别的评估信息')
    
    # 评估相关参数
    parser.add_argument('--N', type=str, default='N',
                        help='粒度级别标识符，用于选择合适的参考时间线版本。"N"表示使用原始完整时间线，数字如"5"、"10"表示使用对应粒度的meta_timeline')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='蕴含判断阈值 (0-1), 越高越严格, 默认: 0.3')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='每批次处理的请求数量，值越大并行度越高，默认: 10')
    parser.add_argument('--force_reevaluate', action='store_true',
                        help='强制重新评估所有主题，即使已有评估结果')
    parser.add_argument('--clear_detail_file', action='store_true',
                        help='在开始评估前清空节点详细信息文件')
    parser.add_argument('--fact_time_window', type=int, default=60,
                        help='事实性评估时的时间窗口（天），默认60')
    parser.add_argument('--save_detail', action='store_true',
                        help='是否保存节点详细信息，默认保存。如需关闭请不加该参数')
    
    return parser.parse_args()

if __name__ == "__main__":
    import argparse
    import time
    import asyncio
    args = parse_args()
    print(f"===== 开始评估 =====")
    print(f"粒度: {args.N}")
    print(f"阈值: {args.threshold}")
    print(f"详细日志将写入: {log_file}")
    print(f"评估结果将保存到: {args.output_file}")
    print(f"详细评估结果将保存到: {args.detail_output_file}")
    print(f"* 注意: 每评估完一个主题就会立即保存节点详细信息，不会等待全部评估完成")
    print(f"* 注意: 信息量计算仅使用召回率（Recall）进行评估，与DGTLS-Bench保持一致")
    print(f"* 注意: 粒度一致性评估逻辑已与DGTLS-Bench同步，会严格检查节点匹配与边连接")
    print("评估中，请稍候...\n")
    start_time = time.time()
    try:
        results, overall_results = asyncio.run(evaluate_timelines(args))
        print(f"\n评估完成！用时: {time.time() - start_time:.2f}秒")
        print(f"平均分: {overall_results['average'] if overall_results else '无'}")
    except Exception as e:
        print(f"\n评估过程中出错: {e}")
