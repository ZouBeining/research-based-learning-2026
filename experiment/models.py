"""
模型定义模块
包含：
1. TF-IDF + Logistic Regression (sklearn)
2. TF-IDF + Linear SVM (sklearn)
3. TextCNN (PyTorch)
4. BiLSTM (PyTorch)
"""

import time
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

# sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.base import BaseEstimator

# PyTorch models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ModelResult:
    """模型评估结果"""
    model_name: str
    dataset_name: str
    accuracy: float
    macro_f1: float
    train_time: float
    inference_time: float
    n_params: int
    per_class_f1: Dict = field(default_factory=dict)
    # 以下字段由实验脚本填充
    dataset_key: str = ''
    language: str = ''
    n_train: int = 0
    n_test: int = 0

    def to_dict(self) -> Dict:
        return {
            'model': self.model_name,
            'dataset': self.dataset_name,
            'dataset_key': self.dataset_key,
            'language': self.language,
            'accuracy': self.accuracy,
            'macro_f1': self.macro_f1,
            'train_time_sec': self.train_time,
            'inference_time_sec': self.inference_time,
            'n_params': self.n_params,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'per_class_f1': self.per_class_f1,
        }


# ==================== sklearn 线性模型 ====================

class LinearModelWrapper:
    """sklearn 线性模型的统一封装"""
    
    def __init__(self, model_name: str, estimator: BaseEstimator):
        self.model_name = model_name
        self.estimator = estimator
        self.is_fitted = False
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        start = time.time()
        self.estimator.fit(X_train, y_train)
        self.train_time = time.time() - start
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        start = time.time()
        preds = self.estimator.predict(X)
        self.inference_time = time.time() - start
        return preds
    
    def predict_proba(self, X):
        """预测概率（用于校准后的 SVM）"""
        if hasattr(self.estimator, 'predict_proba'):
            return self.estimator.predict_proba(X)
        elif hasattr(self.estimator, 'decision_function'):
            # 对于 LinearSVC，使用 decision_function 并通过 softmax 近似
            scores = self.estimator.decision_function(X)
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
            # 简单的 softmax
            exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return None
    
    def evaluate(self, X_test, y_test, label_names=None) -> ModelResult:
        """评估模型"""
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        # 每类 F1
        per_class_f1 = {}
        if label_names:
            report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0)
            for i, name in enumerate(label_names):
                per_class_f1[name] = report.get(name, {}).get('f1-score', 0.0)
        else:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            for i in range(len(set(y_test))):
                per_class_f1[i] = report.get(str(i), {}).get('f1-score', 0.0)
        
        # 参数量估算
        n_params = self._count_params()
        
        return ModelResult(
            model_name=self.model_name,
            dataset_name='',  # 由调用者填充
            accuracy=accuracy,
            macro_f1=macro_f1,
            train_time=self.train_time,
            inference_time=self.inference_time,
            n_params=n_params,
            per_class_f1=per_class_f1
        )
    
    def _count_params(self) -> int:
        """估算参数量"""
        if hasattr(self.estimator, 'coef_'):
            return self.estimator.coef_.size + (self.estimator.intercept_.size if hasattr(self.estimator, 'intercept_') else 0)
        return 0


def create_logistic_regression(C: float = 1.0, max_iter: int = 1000,
                               random_state: int = 42) -> LinearModelWrapper:
    """创建 TF-IDF + Logistic Regression 模型"""
    lr = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs',
    )
    return LinearModelWrapper('TF-IDF + Logistic Regression', lr)


def create_linear_svm(C: float = 1.0, max_iter: int = 5000,
                      random_state: int = 42, calibrated: bool = True) -> LinearModelWrapper:
    """创建 TF-IDF + Linear SVM 模型"""
    svm = LinearSVC(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        dual='auto'
    )
    
    if calibrated:
        # 使用 CalibratedClassifierCV 获取概率输出
        svm = CalibratedClassifierCV(svm, cv=3, method='sigmoid')
    
    return LinearModelWrapper('TF-IDF + Linear SVM', svm)


# ==================== PyTorch 神经网络模型 ====================

class TextCNN(nn.Module):
    """
    TextCNN 文本分类模型
    Kim (2014) - Convolutional Neural Networks for Sentence Classification
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, 
                 num_classes: int = 4, num_filters: int = 100,
                 filter_sizes: Tuple[int, ...] = (3, 4, 5),
                 dropout: float = 0.5, padding_idx: int = 0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        
        # 卷积 + 全局最大池化
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(x))  # (batch, num_filters, seq_len - fs + 1)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # (batch, num_filters)
            conv_outs.append(c)
        
        x = torch.cat(conv_outs, dim=1)  # (batch, num_filters * len(filter_sizes))
        x = self.dropout(x)
        x = self.fc(x)  # (batch, num_classes)
        return x


class BiLSTM(nn.Module):
    """
    BiLSTM 文本分类模型
    双向 LSTM + 注意力机制（可选）
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 1,
                 num_classes: int = 4, dropout: float = 0.5,
                 bidirectional: bool = True, padding_idx: int = 0,
                 use_attention: bool = False):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.use_attention = use_attention
        
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim),
                nn.Tanh(),
                nn.Linear(lstm_output_dim, 1)
            )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        if self.use_attention:
            for layer in self.attention:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim * 2)
        
        if self.use_attention:
            # 注意力池化
            attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
            x = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden_dim * 2)
        else:
            # 取最后一个时间步（双向拼接）
            x = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)
        
        x = self.dropout(x)
        x = self.fc(x)  # (batch, num_classes)
        return x


# ==================== PyTorch 模型训练器 ====================

class TorchTrainer:
    """PyTorch 模型训练与评估封装"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu',
                 lr: float = 1e-3, weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.train_time = 0.0
        self.inference_time = 0.0
    
    def _create_dataloader(self, X, y, batch_size: int = 32, shuffle: bool = True):
        """创建 DataLoader"""
        # X 可以是稀疏矩阵或 dense 数组
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X_tensor = torch.LongTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs: int = 10, batch_size: int = 32,
            patience: int = 3, verbose: bool = True) -> 'TorchTrainer':
        """训练模型"""
        train_loader = self._create_dataloader(X_train, y_train, batch_size, shuffle=True)
        
        if X_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, batch_size, shuffle=False)
        else:
            val_loader = None
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        logits = self.model(batch_X)
                        loss = self.criterion(logits, batch_y)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose:
                    print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose:
                    print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}")
        
        self.train_time = time.time() - start_time
        
        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return self
    
    def predict(self, X, batch_size: int = 64) -> np.ndarray:
        """预测"""
        self.model.eval()
        loader = self._create_dataloader(X, np.zeros(len(X)), batch_size, shuffle=False)
        
        all_preds = []
        start_time = time.time()
        
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                logits = self.model(batch_X)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
        
        self.inference_time = time.time() - start_time
        return np.array(all_preds)
    
    def predict_proba(self, X, batch_size: int = 64) -> np.ndarray:
        """预测概率"""
        self.model.eval()
        loader = self._create_dataloader(X, np.zeros(len(X)), batch_size, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                logits = self.model(batch_X)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs)
        return np.array(all_probs)
    
    def evaluate(self, X_test, y_test, label_names=None) -> ModelResult:
        """评估模型"""
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        # 每类 F1
        per_class_f1 = {}
        if label_names:
            report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0)
            for name in label_names:
                per_class_f1[name] = report.get(name, {}).get('f1-score', 0.0)
        else:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            for i in range(len(set(y_test))):
                per_class_f1[i] = report.get(str(i), {}).get('f1-score', 0.0)
        
        # 参数量
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return ModelResult(
            model_name=self.model.__class__.__name__,
            dataset_name='',
            accuracy=accuracy,
            macro_f1=macro_f1,
            train_time=self.train_time,
            inference_time=self.inference_time,
            n_params=n_params,
            per_class_f1=per_class_f1
        )
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def create_textcnn(vocab_size: int, num_classes: int, 
                   embed_dim: int = 128, num_filters: int = 100,
                   filter_sizes: Tuple[int, ...] = (3, 4, 5),
                   dropout: float = 0.5, device: str = 'cpu') -> Tuple[TextCNN, TorchTrainer]:
    """创建 TextCNN 模型和训练器"""
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_filters=num_filters,
        filter_sizes=filter_sizes,
        dropout=dropout
    )
    trainer = TorchTrainer(model, device=device)
    return model, trainer


def create_bilstm(vocab_size: int, num_classes: int,
                  embed_dim: int = 128, hidden_dim: int = 128,
                  num_layers: int = 1, dropout: float = 0.5,
                  bidirectional: bool = True, use_attention: bool = False,
                  device: str = 'cpu') -> Tuple[BiLSTM, TorchTrainer]:
    """创建 BiLSTM 模型和训练器"""
    model = BiLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
        use_attention=use_attention
    )
    trainer = TorchTrainer(model, device=device)
    return model, trainer


# ==================== 统一模型工厂 ====================

def create_all_models(vocab_size: int, num_classes: int, 
                      device: str = 'cpu',
                      config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    创建所有对比模型
    返回字典：model_name -> (model_object, trainer_or_wrapper)
    """
    if config is None:
        config = {}
    
    models = {}
    
    # 1. TF-IDF + Logistic Regression
    models['TF-IDF + Logistic Regression'] = create_logistic_regression(
        C=config.get('lr_C', 1.0),
        max_iter=config.get('lr_max_iter', 1000),
        random_state=config.get('random_state', 42)
    )
    
    # 2. TF-IDF + Linear SVM
    models['TF-IDF + Linear SVM'] = create_linear_svm(
        C=config.get('svm_C', 1.0),
        max_iter=config.get('svm_max_iter', 5000),
        random_state=config.get('random_state', 42),
        calibrated=config.get('svm_calibrated', True)
    )
    
    # 3. TextCNN
    cnn_model, cnn_trainer = create_textcnn(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embed_dim=config.get('cnn_embed_dim', 128),
        num_filters=config.get('cnn_num_filters', 100),
        filter_sizes=config.get('cnn_filter_sizes', (3, 4, 5)),
        dropout=config.get('cnn_dropout', 0.5),
        device=device
    )
    models['TextCNN'] = (cnn_model, cnn_trainer)
    
    # 4. BiLSTM
    lstm_model, lstm_trainer = create_bilstm(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embed_dim=config.get('lstm_embed_dim', 128),
        hidden_dim=config.get('lstm_hidden_dim', 128),
        num_layers=config.get('lstm_num_layers', 1),
        dropout=config.get('lstm_dropout', 0.5),
        bidirectional=config.get('lstm_bidirectional', True),
        use_attention=config.get('lstm_attention', False),
        device=device
    )
    models['BiLSTM'] = (lstm_model, lstm_trainer)
    
    return models


if __name__ == '__main__':
    # 简单测试
    print("测试模型创建...")
    
    # 测试 sklearn 模型
    lr_model = create_logistic_regression()
    svm_model = create_linear_svm()
    print(f"LR: {lr_model.model_name}")
    print(f"SVM: {svm_model.model_name}")
    
    # 测试 PyTorch 模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    cnn_model, cnn_trainer = create_textcnn(vocab_size=1000, num_classes=4, device=device)
    print(f"TextCNN params: {cnn_trainer.count_params()}")
    
    lstm_model, lstm_trainer = create_bilstm(vocab_size=1000, num_classes=4, device=device)
    print(f"BiLSTM params: {lstm_trainer.count_params()}")
    
    print("\n模型创建测试通过！")