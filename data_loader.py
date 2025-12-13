import os
import re
from typing import List, Optional
import json


class TextDataLoader:
    """
    支持多种格式的文本数据加载器
    
    支持的文件格式：
    - .txt: 纯文本文件，每行一个样本或段落
    - .json: JSON格式，支持多种结构
    - .jsonl: JSON Lines格式，每行一个JSON对象
    """
    
    def __init__(self, min_length=10, max_length=1000, lower=False, line_mode=False):
        self.min_length = min_length
        self.max_length = max_length
        self.lower = lower
        self.line_mode = line_mode
        
    def clean_text(self, text):
        """清理文本"""
        text = text.strip()
        
        if self.lower:
            text = text.lower()
            
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}\'\"-]', '', text)
        
        return text
    
    def load_txt(self, file_path: str, encoding='utf-8') -> List[str]:
        """
        加载 .txt 文件
        
        格式要求：
        1. 如果 line_mode=True：每行一个样本（适合大数据集）
        2. 如果 line_mode=False：空行分隔段落
        3. 支持 UTF-8 编码
        """
        texts = []
        try:
            if self.line_mode:
                print(f"  Loading in line mode (each line = one sample)...")
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if line:
                            text = self.clean_text(line)
                            if self.min_length <= len(text) <= self.max_length:
                                texts.append(text)
                            
                            if (line_num + 1) % 100000 == 0:
                                print(f"    Processed {line_num + 1:,} lines, loaded {len(texts):,} samples...")
                
                print(f"  Finished: processed {line_num + 1:,} lines, loaded {len(texts):,} valid samples")
            else:
                print(f"  Loading in paragraph mode (empty lines separate paragraphs)...")
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                    
                lines = content.split('\n')
                current_paragraph = []
                total_paragraphs_created = 0
                filtered_count = 0
                
                for line in lines:
                    line = line.strip()
                    if line:
                        current_paragraph.append(line)
                    else:
                        if current_paragraph:
                            text = ' '.join(current_paragraph)
                            total_paragraphs_created += 1
                            text = self.clean_text(text)
                            if self.min_length <= len(text) <= self.max_length:
                                texts.append(text)
                            else:
                                filtered_count += 1
                                if filtered_count <= 5:
                                    print(f"    Filtered paragraph (length {len(text)}): {text[:50]}...")
                            current_paragraph = []
                
                if current_paragraph:
                    text = ' '.join(current_paragraph)
                    total_paragraphs_created += 1
                    text = self.clean_text(text)
                    if self.min_length <= len(text) <= self.max_length:
                        texts.append(text)
                    else:
                        filtered_count += 1
                
                print(f"  Finished: created {total_paragraphs_created} paragraphs, loaded {len(texts)} valid samples")
                if filtered_count > 0:
                    print(f"  Warning: {filtered_count} paragraphs filtered due to length limits (min={self.min_length}, max={self.max_length})")
                    if total_paragraphs_created < len(lines) * 0.1:
                        print(f"  ⚠️  Very few paragraphs created ({total_paragraphs_created}) compared to total lines ({len(lines):,})")
                        print(f"  💡  Recommendation: Use --line_mode to treat each line as a separate sample")
                    
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            import traceback
            traceback.print_exc()
            
        return texts
    
    def load_json(self, file_path: str, text_key: str = 'text') -> List[str]:
        """
        加载 .json 文件
        
        格式要求：
        1. 单个JSON对象，包含文本数组：
           {"texts": ["text1", "text2", ...]}
        
        2. JSON数组：
           [{"text": "text1"}, {"text": "text2"}, ...]
           
        3. 使用 text_key 参数指定文本字段名
        """
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and text_key in item:
                        text = self.clean_text(item[text_key])
                        if self.min_length <= len(text) <= self.max_length:
                            texts.append(text)
                    elif isinstance(item, str):
                        text = self.clean_text(item)
                        if self.min_length <= len(text) <= self.max_length:
                            texts.append(text)
            elif isinstance(data, dict):
                if 'texts' in data and isinstance(data['texts'], list):
                    for text in data['texts']:
                        text = self.clean_text(text)
                        if self.min_length <= len(text) <= self.max_length:
                            texts.append(text)
                elif text_key in data:
                    text = self.clean_text(data[text_key])
                    if self.min_length <= len(text) <= self.max_length:
                        texts.append(text)
                        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
        return texts
    
    def load_jsonl(self, file_path: str, text_key: str = 'text') -> List[str]:
        """
        加载 .jsonl 文件（JSON Lines）
        
        格式要求：
        每行一个JSON对象：
        {"text": "text1"}
        {"text": "text2"}
        ...
        """
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict) and text_key in item:
                            text = self.clean_text(item[text_key])
                            if self.min_length <= len(text) <= self.max_length:
                                texts.append(text)
                        elif isinstance(item, str):
                            text = self.clean_text(item)
                            if self.min_length <= len(text) <= self.max_length:
                                texts.append(text)
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
        return texts
    
    def load_directory(self, directory_path: str, extensions=['.txt', '.json', '.jsonl']) -> List[str]:
        """
        加载目录中的所有文本文件
        
        支持递归搜索子目录
        """
        all_texts = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file_path)
                
                if ext.lower() not in extensions:
                    continue
                
                print(f"Loading {file_path}...")
                
                if ext.lower() == '.txt':
                    texts = self.load_txt(file_path)
                elif ext.lower() == '.json':
                    texts = self.load_json(file_path)
                elif ext.lower() == '.jsonl':
                    texts = self.load_jsonl(file_path)
                else:
                    continue
                
                all_texts.extend(texts)
                print(f"  Total loaded: {len(texts)} texts from {file}")
        
        return all_texts
    
    def load_file(self, file_path: str, text_key: str = 'text') -> List[str]:
        """
        自动识别文件类型并加载
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.txt':
            return self.load_txt(file_path)
        elif ext == '.json':
            return self.load_json(file_path, text_key)
        elif ext == '.jsonl':
            return self.load_jsonl(file_path, text_key)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
