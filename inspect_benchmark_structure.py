"""
简易脚本：查看benchmark JSON文件结构
"""

import json
import sys

def inspect_json_structure(file_path, max_items=3):
    """
    查看JSON文件结构，只读取前几条数据
    
    Args:
        file_path: JSON文件路径
        max_items: 最多读取的条目数
    """
    print(f"正在检查文件: {file_path}\n")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 尝试加载JSON
            data = json.load(f)
        
        # 检查数据类型
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"字典的键: {list(data.keys())}\n")
            
            # 如果是字典，查看每个键的结构
            for key in data.keys():
                print(f"=== 键: {key} ===")
                value = data[key]
                print(f"值类型: {type(value)}")
                
                if isinstance(value, list):
                    print(f"列表长度: {len(value)}")
                    if len(value) > 0:
                        print(f"\n第一个元素的类型: {type(value[0])}")
                        if isinstance(value[0], dict):
                            print(f"第一个元素的键: {list(value[0].keys())}")
                            print(f"\n第一个元素的示例数据:")
                            for k, v in value[0].items():
                                if k.endswith('b64') or 'image' in k.lower():
                                    # 对于base64字段，只显示前50个字符
                                    if isinstance(v, str) and len(v) > 50:
                                        print(f"  {k}: {v[:50]}... (长度: {len(v)})")
                                    else:
                                        print(f"  {k}: {v}")
                                else:
                                    print(f"  {k}: {v}")
                        else:
                            print(f"第一个元素: {value[0]}")
                        
                        # 显示前几个元素
                        print(f"\n前{min(max_items, len(value))}个元素的概览:")
                        for i, item in enumerate(value[:max_items]):
                            if isinstance(item, dict):
                                print(f"\n--- 元素 {i+1} ---")
                                for k, v in item.items():
                                    if k.endswith('b64') or 'image' in k.lower():
                                        if isinstance(v, str) and len(v) > 50:
                                            print(f"  {k}: [base64 data, 长度: {len(v)}]")
                                        else:
                                            print(f"  {k}: {v}")
                                    else:
                                        print(f"  {k}: {v}")
                print("\n" + "="*60 + "\n")
        
        elif isinstance(data, list):
            print(f"列表长度: {len(data)}")
            if len(data) > 0:
                print(f"\n第一个元素的类型: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"第一个元素的键: {list(data[0].keys())}")
                    print(f"\n前{min(max_items, len(data))}个元素的示例:")
                    for i, item in enumerate(data[:max_items]):
                        print(f"\n--- 元素 {i+1} ---")
                        for k, v in item.items():
                            if k.endswith('b64') or 'image' in k.lower():
                                if isinstance(v, str) and len(v) > 50:
                                    print(f"  {k}: [base64 data, 长度: {len(v)}]")
                                else:
                                    print(f"  {k}: {v}")
                            else:
                                print(f"  {k}: {v}")
        
        # 统计subset分布
        print("\n" + "="*60)
        print("统计 subset 分布:")
        print("="*60)
        subset_count = {}
        
        if isinstance(data, dict):
            # 尝试在字典的值中找列表
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and 'subset' in item:
                            subset = item.get('subset', 'unknown')
                            subset_count[subset] = subset_count.get(subset, 0) + 1
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'subset' in item:
                    subset = item.get('subset', 'unknown')
                    subset_count[subset] = subset_count.get(subset, 0) + 1
        
        if subset_count:
            print("\nSubset分布:")
            for subset, count in sorted(subset_count.items()):
                print(f"  {subset}: {count} 条")
            print(f"  总计: {sum(subset_count.values())} 条")
        else:
            print("未找到 'subset' 字段")
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    file_path = "/data2/yixuan/Benchmark/version_2_50_pair/version_2_with_imagesb64.json"
    inspect_json_structure(file_path, max_items=2)


