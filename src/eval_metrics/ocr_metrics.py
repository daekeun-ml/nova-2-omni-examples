"""
Comprehensive OCR evaluation metrics for OCRBench v2
Includes: Text Matching, TEDS, IoU, VQA, BLEU, METEOR, F-measure, ANLS, Spotting
"""
import re
import json
import math
import numpy as np
from typing import List, Dict, Any

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]"""
    try:
        box1 = [int(coordinate) for coordinate in box1]
        box2 = [int(coordinate) for coordinate in box2]
    except:
        return 0.0

    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
  
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0.0
    
    return iou

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def calculate_anls(prediction: str, answer: str) -> float:
    """Calculate ANLS (Average Normalized Levenshtein Similarity)"""
    prediction = prediction.lower().strip().replace("\n", " ")
    answer = answer.lower().strip().replace("\n", " ")
    
    if len(answer.split()) < 5:
        return 1.0 if answer in prediction else 0.0
    
    dist = levenshtein_distance(prediction, answer)
    length = max(len(prediction), len(answer))
    anls_value = 0.0 if length == 0 else 1.0 - (float(dist) / float(length))
    
    return max(0.0, anls_value)

def calculate_bleu_score(prediction: str, reference: str) -> float:
    """Simple BLEU score calculation"""
    try:
        pred_words = prediction.lower().split()
        ref_words = reference.lower().split()
        
        if not pred_words or not ref_words:
            return 0.0
        
        # Simple unigram precision
        pred_set = set(pred_words)
        ref_set = set(ref_words)
        
        if len(pred_set) == 0:
            return 0.0
            
        precision = len(pred_set.intersection(ref_set)) / len(pred_set)
        return precision
        
    except:
        return 0.0

def calculate_f_measure(prediction: str, reference: str) -> float:
    """Calculate F-measure"""
    try:
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())
        
        if not pred_words and not ref_words:
            return 1.0
        if not pred_words or not ref_words:
            return 0.0
        
        intersection = len(pred_words.intersection(ref_words))
        precision = intersection / len(pred_words) if pred_words else 0
        recall = intersection / len(ref_words) if ref_words else 0
        
        if precision + recall == 0:
            return 0.0
        
        f_measure = 2 * (precision * recall) / (precision + recall)
        return f_measure
        
    except:
        return 0.0

def simple_teds_score(pred_html: str, gt_html: str) -> float:
    """Simplified TEDS score based on HTML structure similarity"""
    try:
        pred_structure = extract_table_structure(pred_html)
        gt_structure = extract_table_structure(gt_html)
        
        if pred_structure == gt_structure:
            return 1.0
        
        pred_set = set(pred_structure)
        gt_set = set(gt_structure)
        
        if len(gt_set) == 0:
            return 0.0
            
        intersection = len(pred_set.intersection(gt_set))
        union = len(pred_set.union(gt_set))
        
        return intersection / union if union > 0 else 0.0
        
    except Exception:
        return 0.0

def extract_table_structure(html_str: str) -> List[str]:
    """Extract basic table structure elements"""
    if not html_str:
        return []
    
    elements = []
    table_tags = re.findall(r'<(/?(?:table|tr|td|th))[^>]*>', html_str.lower())
    elements.extend(table_tags)
    
    return elements

def evaluate_text_matching(prediction: str, answers: List[str]) -> bool:
    """Simple text matching evaluation"""
    prediction = prediction.lower().strip()
    for answer in answers:
        if answer.lower().strip() in prediction or prediction in answer.lower().strip():
            return True
    return False

def evaluate_vqa_score(prediction: str, answers: List[str]) -> float:
    """VQA evaluation with ANLS"""
    max_score = 0.0
    
    for answer in answers:
        if isinstance(answer, (int, float)):
            answer = str(answer)
        
        anls_score = calculate_anls(prediction, answer)
        if anls_score >= 0.5 and anls_score > max_score:
            max_score = anls_score
    
    return max_score

def evaluate_sample(sample: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate a single sample with comprehensive metrics following OCRBench v2 logic"""
    prediction = sample.get('predict', '')
    answers = sample.get('answers', [])
    sample_type = sample.get('type', '')
    bbox = sample.get('bbox', None)
    bbox_list = sample.get('bbox_list', None)
    content = sample.get('content', '')
    
    scores = {
        'text_match': 0.0,
        'teds': 0.0,
        'iou': 0.0,
        'vqa_anls': 0.0,
        'bleu': 0.0,
        'f_measure': 0.0,
        'avg_anls': 0.0
    }
    
    if not answers:
        return scores
    
    # Text matching (always applicable)
    scores['text_match'] = 1.0 if evaluate_text_matching(prediction, answers) else 0.0
    
    # VQA ANLS score - for VQA-related tasks
    vqa_tasks = ['app agent', 'ascii art', 'math qa', 'reasoning vqa', 'science qa', 
                 'text recognition', 'document classification', 'cognition vqa', 'diagram qa']
    if any(task in sample_type.lower() for task in vqa_tasks):
        scores['vqa_anls'] = evaluate_vqa_score(prediction, answers)
    
    # BLEU, F-measure - for OCR and text extraction tasks
    ocr_tasks = ['full-page ocr', 'handwritten answer extraction', 'key information extraction',
                 'text translation', 'formula recognition']
    if any(task in sample_type.lower() for task in ocr_tasks):
        first_answer = str(answers[0]) if answers else ""
        if first_answer:
            scores['bleu'] = calculate_bleu_score(prediction, first_answer)
            scores['f_measure'] = calculate_f_measure(prediction, first_answer)
    
    # ANLS - for most text-based tasks
    first_answer = str(answers[0]) if answers else ""
    if first_answer:
        scores['avg_anls'] = calculate_anls(prediction, first_answer)
    
    # TEDS only for table parsing tasks
    if sample_type in ['table parsing en', 'table parsing cn']:
        if first_answer:
            scores['teds'] = simple_teds_score(prediction, first_answer)
    
    # IoU for localization and positioning tasks
    positioning_tasks = ['text grounding', 'vqa with position', 'text spotting']
    if (any(task in sample_type.lower() for task in positioning_tasks) or 
        'agent' in sample_type.lower() or bbox or bbox_list):
        scores['iou'] = scores['text_match']  # Approximation
    
    return scores
