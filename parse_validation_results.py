#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量推理结果解析脚本
用法: python parse_validation_results.py <result_json_file>
"""

import json
import sys
import argparse
from collections import Counter
from typing import List, Dict

def load_results(file_path: str) -> List[Dict]:
    """加载结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def basic_statistics(results: List[Dict]) -> Dict:
    """基本统计信息"""
    total = len(results)
    if total == 0:
        return {
            'total_images': 0,
            'avg_caption_length': 0,
            'top_words': []
        }
    
    avg_length = sum(len(r['prediction'].split()) for r in results) / total
    
    # 词频统计
    all_words = []
    for r in results:
        all_words.extend(r['prediction'].lower().split())
    word_freq = Counter(all_words)
    
    return {
        'total_images': total,
        'avg_caption_length': avg_length,
        'top_words': word_freq.most_common(20)
    }

def compare_results(file1: str, file2: str):
    """对比两个结果文件"""
    results1 = {r['image_name']: r for r in load_results(file1)}
    results2 = {r['image_name']: r for r in load_results(file2)}
    
    common = set(results1.keys()) & set(results2.keys())
    different = []
    identical = []
    
    for img_name in common:
        pred1 = results1[img_name]['prediction']
        pred2 = results2[img_name]['prediction']
        if pred1 != pred2:
            different.append({
                'image': img_name,
                'file1': pred1,
                'file2': pred2
            })
        else:
            identical.append(img_name)
    
    return {
        'common_images': len(common),
        'different': len(different),
        'identical': len(identical),
        'different_samples': different[:10]  # 前10个不同的样本
    }

def export_for_evaluation(results: List[Dict], output_file: str):
    """导出为 COCO 评估格式"""
    coco_format = []
    for r in results:
        coco_format.append({
            'image_id': r['image_name'],
            'caption': r['prediction']
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=2, ensure_ascii=False)
    
    print(f"已导出到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='解析批量推理结果')
    parser.add_argument('result_file', type=str, help='结果 JSON 文件路径')
    parser.add_argument('--compare', type=str, help='对比的另一个结果文件')
    parser.add_argument('--export', type=str, help='导出为评估格式的文件路径')
    parser.add_argument('--show-samples', type=int, default=5, help='显示样本数量')
    
    args = parser.parse_args()
    
    # 加载结果
    print(f"加载结果文件: {args.result_file}")
    try:
        results = load_results(args.result_file)
    except FileNotFoundError:
        print(f"错误: 文件不存在: {args.result_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: JSON 格式错误: {e}")
        sys.exit(1)
    
    # 基本统计
    print("\n=== 基本统计 ===")
    stats = basic_statistics(results)
    print(f"总图像数: {stats['total_images']}")
    print(f"平均描述长度: {stats['avg_caption_length']:.2f} 词")
    print(f"\n最常见的20个词:")
    for word, count in stats['top_words']:
        print(f"  {word}: {count}")
    
    # 显示样本
    print(f"\n=== 前 {args.show_samples} 个样本 ===")
    for i, r in enumerate(results[:args.show_samples]):
        print(f"\n图像 {i+1}: {r['image_name']}")
        print(f"预测: {r['prediction']}")
        print(f"真实标注数: {len(r['captions'])}")
        if r['captions']:
            print(f"示例标注: {r['captions'][0]}")
    
    # 对比分析
    if args.compare:
        print(f"\n=== 对比分析 ===")
        print(f"对比文件: {args.compare}")
        try:
            comparison = compare_results(args.result_file, args.compare)
            print(f"共同图像数: {comparison['common_images']}")
            print(f"预测相同: {comparison['identical']}")
            print(f"预测不同: {comparison['different']}")
            
            if comparison['different_samples']:
                print(f"\n前10个不同的预测:")
                for diff in comparison['different_samples']:
                    print(f"\n图像: {diff['image']}")
                    print(f"  文件1: {diff['file1']}")
                    print(f"  文件2: {diff['file2']}")
        except FileNotFoundError:
            print(f"错误: 对比文件不存在: {args.compare}")
        except Exception as e:
            print(f"错误: {e}")
    
    # 导出
    if args.export:
        print(f"\n=== 导出 ===")
        export_for_evaluation(results, args.export)

if __name__ == '__main__':
    main()

