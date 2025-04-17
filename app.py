from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

app = Flask(__name__)

# 加载模型和tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 创建模型字典
models = {
    'neutral': BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=4),
    'male': BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=4),
    'female': BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=4)
}

# 加载训练好的模型权重
model_files = {
    'neutral': 'emotion_model.pth',
    'male': 'emotion_model_male.pth',
    'female': 'emotion_model_female.pth'
}

for model_type, model in models.items():
    try:
        model.load_state_dict(torch.load(model_files[model_type], map_location=device))
        print(f"成功加载{model_type}情感分析模型")
    except:
        print(f"未找到预训练{model_type}情感模型，使用初始模型")
    model.to(device)
    model.eval()

# 情感标签映射
emotion_labels = {
    0: {
        'emotion': 'neutral',
        'chinese': '平静',
        'description': '您现在看起来很平静，处于一个比较客观理性的状态。'
    },
    1: {
        'emotion': 'happy',
        'chinese': '快乐',
        'description': '从您的话语中，我能感受到愉悦和积极的情绪。'
    },
    2: {
        'emotion': 'sad',
        'chinese': '悲伤',
        'description': '您似乎有些低落或伤心，需要我陪您聊聊吗？'
    },
    3: {
        'emotion': 'angry',
        'chinese': '愤怒',
        'description': '您现在似乎很生气或不满，让我们一起冷静下来谈谈。'
    }
}

def analyze_emotion(text, model_type='neutral'):
    """分析文本的情感状态"""
    # 获取选定的模型
    model = models.get(model_type, models['neutral'])
    
    # 对文本进行编码
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 将输入移到正确的设备上
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 进行预测
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    # 获取预测结果
    predicted_emotion = torch.argmax(predictions).item()
    confidence = predictions[0][predicted_emotion].item()
    
    # 获取所有情感的概率分布
    emotion_probs = {
        emotion_labels[i]['chinese']: round(prob.item() * 100, 2)
        for i, prob in enumerate(predictions[0])
    }
    
    return {
        'main_emotion': emotion_labels[predicted_emotion],
        'confidence': round(confidence * 100, 2),
        'emotion_distribution': emotion_probs
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('context', '').strip()
    model_type = data.get('model_type', 'neutral')
    
    if not user_input:
        return jsonify({
            'response': '请输入一些文字，让我来分析您的情感状态。',
            'emotion': 'neutral',
            'analysis': None
        })
    
    # 分析情感
    analysis = analyze_emotion(user_input, model_type)
    main_emotion = analysis['main_emotion']
    
    # 生成回复
    model_type_names = {
        'neutral': '通用视角',
        'male': '男性视角',
        'female': '女性视角'
    }
    
    response = f"[{model_type_names[model_type]}分析结果]\n"
    response += f"{main_emotion['description']} (置信度: {analysis['confidence']}%)\n\n"
    response += "情感分布分析：\n"
    for emotion, prob in analysis['emotion_distribution'].items():
        response += f"- {emotion}: {prob}%\n"
    
    return jsonify({
        'response': response,
        'emotion': main_emotion['emotion'],
        'analysis': analysis
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001) 