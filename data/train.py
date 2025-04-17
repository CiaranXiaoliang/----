import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import os
import random
import jieba
from collections import Counter
import torch.nn.functional as F
import requests
import zipfile
import io
import json
from torch.cuda.amp import autocast, GradScaler  # 添加混合精度训练支持

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    # 设置 CUDA 优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # 设置默认设备
    torch.cuda.set_device(0)
    # 清空 GPU 缓存
    torch.cuda.empty_cache()

# 扩展同义词词典，添加性别特定的词汇
SYNONYMS = {
    '开心': ['高兴', '愉快', '快乐', '欢喜', '兴奋', '欣喜', '愉悦', '舒畅'],
    '伤心': ['难过', '悲伤', '哀伤', '悲痛', '伤心', '忧郁', '沮丧', '失落'],
    '生气': ['愤怒', '恼火', '气愤', '发怒', '暴怒', '震怒', '气愤', '恼羞成怒'],
    '平静': ['平和', '安宁', '安静', '宁静', '淡定', '从容', '平和', '心平气和']
}

# 性别特定的情感表达词典
GENDER_SPECIFIC_EXPRESSIONS = {
    'male': {
        '开心': ['爽', '痛快', '给力', '牛逼', '厉害', '帅', '酷'],
        '伤心': ['郁闷', '憋屈', '不爽', '难受', '窝火', '憋闷'],
        '生气': ['火大', '不爽', '操蛋', '扯淡', '坑爹', '坑人'],
        '平静': ['还行', '凑合', '一般', '马马虎虎', '过得去']
    },
    'female': {
        '开心': ['好开心', '好高兴', '好喜欢', '好棒', '好可爱', '好美'],
        '伤心': ['好难过', '好伤心', '好委屈', '好想哭', '好失望'],
        '生气': ['好生气', '好讨厌', '好烦', '好讨厌', '好烦人'],
        '平静': ['还好', '还行', '一般般', '还可以', '过得去']
    }
}

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, augment=False, gender=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.gender = gender
        if augment:
            self.word_freq = self._build_word_frequency()

    def _build_word_frequency(self):
        """构建词频统计"""
        all_words = []
        for text in self.texts:
            # 确保文本是字符串类型
            text = str(text).strip()
            if text:  # 只处理非空文本
                words = jieba.lcut(text)
                all_words.extend(words)
        return Counter(all_words)

    def _augment_text(self, text):
        """数据增强：同义词替换和随机插入"""
        # 确保文本是字符串类型
        text = str(text).strip()
        if not text:
            return text
            
        words = jieba.lcut(text)
        new_words = []
        
        for word in words:
            if random.random() < 0.3:  # 30%的概率进行增强
                # 同义词替换
                if word in SYNONYMS:
                    new_word = random.choice(SYNONYMS[word])
                    new_words.append(new_word)
                # 性别特定的表达替换
                elif self.gender and self.gender != 'neutral' and word in GENDER_SPECIFIC_EXPRESSIONS[self.gender]:
                    new_word = random.choice(GENDER_SPECIFIC_EXPRESSIONS[self.gender][word])
                    new_words.append(new_word)
                # 随机插入
                elif random.random() < 0.1:  # 10%的概率插入随机词
                    new_words.append(word)
                    if self.gender and self.gender != 'neutral':
                        # 插入性别特定的语气词
                        if self.gender == 'male':
                            new_words.append(random.choice(['啊', '嗯', '哦', '哈']))
                        else:
                            new_words.append(random.choice(['啦', '呢', '呀', '嘛']))
                    new_words.append(random.choice(list(self.word_freq.keys())))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        return ''.join(new_words)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).strip()
        if self.augment and random.random() < 0.5:  # 50%的概率进行数据增强
            text = self._augment_text(text)

        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_data(data_path, gender=None):
    """加载和预处理数据"""
    df = pd.read_csv(data_path)
    
    # 如果指定了性别，则筛选对应性别的数据
    if gender:
        df = df[df['gender'] == gender]
    
    texts = df['text'].values
    labels = df['label'].values
    
    return texts, labels

class EmotionClassifier(torch.nn.Module):
    def __init__(self, model_name='bert-base-chinese', num_labels=4):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

def train_model(model_type, train_texts, train_labels, val_texts, val_labels, 
                num_epochs=10, batch_size=64, learning_rate=1e-5, warmup_steps=100):
    """训练模型"""
    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = EmotionClassifier()
    model.to(device)  # 将模型移到 GPU

    # 创建数据集和数据加载器
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, augment=True, gender=model_type)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, gender=model_type)
    
    # 使用 DataLoader 的 num_workers 参数加速数据加载
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=4, pin_memory=True if torch.cuda.is_available() else False)

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 训练循环
    best_val_f1 = 0
    early_stopping_patience = 5
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # 计算训练指标
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"训练损失: {train_loss/len(train_loader):.4f}")
        print(f"训练F1分数: {train_f1:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        # 计算验证指标
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        print(f"验证损失: {val_loss/len(val_loader):.4f}")
        print(f"验证F1分数: {val_f1:.4f}")
        print("\n分类报告:")
        print(classification_report(all_val_labels, all_val_preds))

        # 早停和模型保存
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("早停触发，停止训练")
                break

    # 保存最佳模型
    if best_model_state is not None:
        model_save_path = f'emotion_model_{model_type}.pth'
        torch.save(best_model_state, model_save_path)
        print(f"保存最佳模型到 {model_save_path}")

def download_and_process_datasets():
    """下载和处理公开数据集"""
    datasets = []
    
    # 1. ChnSentiCorp数据集
    print("下载ChnSentiCorp数据集...")
    url = "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"
    try:
        response = requests.get(url)
        df = pd.read_csv(io.StringIO(response.text))
        # 确保列名正确
        if 'review' in df.columns:
            df = df.rename(columns={'review': 'text'})
        df['label'] = df['label'].map({1: 1, 0: 2})  # 1: 正面(开心), 0: 负面(伤心)
        df['gender'] = 'neutral'  # 通用数据集
        datasets.append(df)
        print(f"成功下载ChnSentiCorp数据集，包含{len(df)}条数据")
    except Exception as e:
        print(f"下载ChnSentiCorp数据集失败: {e}")

    # 2. Weibo-Senti100k数据集
    print("下载Weibo-Senti100k数据集...")
    url = "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/weibo_senti_100k/weibo_senti_100k.csv"
    try:
        response = requests.get(url)
        df = pd.read_csv(io.StringIO(response.text))
        # 确保列名正确
        if 'review' in df.columns:
            df = df.rename(columns={'review': 'text'})
        df['label'] = df['label'].map({1: 1, 0: 2})  # 1: 正面(开心), 0: 负面(伤心)
        df['gender'] = 'neutral'  # 通用数据集
        datasets.append(df)
        print(f"成功下载Weibo-Senti100k数据集，包含{len(df)}条数据")
    except Exception as e:
        print(f"下载Weibo-Senti100k数据集失败: {e}")

    # 3. NLPCC2014数据集
    print("下载NLPCC2014数据集...")
    url = "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/nlpcc2014/nlpcc2014.csv"
    try:
        response = requests.get(url)
        df = pd.read_csv(io.StringIO(response.text))
        # 确保列名正确
        if 'review' in df.columns:
            df = df.rename(columns={'review': 'text'})
        df['label'] = df['label'].map({1: 1, 0: 2})  # 1: 正面(开心), 0: 负面(伤心)
        df['gender'] = 'neutral'  # 通用数据集
        datasets.append(df)
        print(f"成功下载NLPCC2014数据集，包含{len(df)}条数据")
    except Exception as e:
        print(f"下载NLPCC2014数据集失败: {e}")

    # 合并所有数据集
    if datasets:
        combined_df = pd.concat(datasets, ignore_index=True)
        print(f"合并后的数据集包含{len(combined_df)}条数据")
        
        # 保存合并后的数据集
        combined_df.to_csv('combined_emotion_data.csv', index=False)
        print("数据集已保存到 combined_emotion_data.csv")
        
        return combined_df
    else:
        raise Exception("未能下载任何数据集")

def prepare_training_data():
    """准备训练数据"""
    try:
        # 尝试加载本地数据集
        df = pd.read_csv('combined_emotion_data.csv')
        print("使用本地数据集")
        
        # 检查并重命名列
        if 'review' in df.columns:
            df = df.rename(columns={'review': 'text'})
        elif 'content' in df.columns:
            df = df.rename(columns={'content': 'text'})
        elif 'sentence' in df.columns:
            df = df.rename(columns={'sentence': 'text'})
            
        # 确保必要的列存在
        if 'text' not in df.columns:
            raise ValueError("数据集中没有找到文本列")
            
        if 'label' not in df.columns:
            raise ValueError("数据集中没有找到标签列")
            
        if 'gender' not in df.columns:
            df['gender'] = 'neutral'
            
        # 处理文本数据，确保所有文本都是字符串类型
        df['text'] = df['text'].astype(str)
        # 移除空文本
        df = df[df['text'].str.strip() != '']
            
    except Exception as e:
        print(f"加载本地数据集失败: {e}")
        print("开始下载新的数据集...")
        df = download_and_process_datasets()
    
    # 为男性和女性生成增强数据
    male_df = df.copy()
    male_df['gender'] = 'male'
    male_df['text'] = male_df['text'].apply(lambda x: str(x) + random.choice(['啊', '嗯', '哦', '哈']))
    
    female_df = df.copy()
    female_df['gender'] = 'female'
    female_df['text'] = female_df['text'].apply(lambda x: str(x) + random.choice(['啦', '呢', '呀', '嘛']))
    
    # 合并所有数据
    final_df = pd.concat([df, male_df, female_df], ignore_index=True)
    
    # 保存最终数据集
    final_df.to_csv('final_emotion_data.csv', index=False)
    print(f"最终数据集包含{len(final_df)}条数据")
    
    return final_df

def main():
    # 准备训练数据
    print("准备训练数据...")
    df = prepare_training_data()
    
    # 训练通用模型
    print("\n训练通用模型...")
    neutral_df = df[df['gender'] == 'neutral']
    texts, labels = neutral_df['text'].values, neutral_df['label'].values
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    train_model('neutral', train_texts, train_labels, val_texts, val_labels)

    # 训练男性视角模型
    print("\n训练男性视角模型...")
    male_df = df[df['gender'] == 'male']
    texts, labels = male_df['text'].values, male_df['label'].values
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    train_model('male', train_texts, train_labels, val_texts, val_labels)

    # 训练女性视角模型
    print("\n训练女性视角模型...")
    female_df = df[df['gender'] == 'female']
    texts, labels = female_df['text'].values, female_df['label'].values
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    train_model('female', train_texts, train_labels, val_texts, val_labels)

if __name__ == '__main__':
    main() 