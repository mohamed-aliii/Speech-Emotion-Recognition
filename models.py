"""
Model Architecture Definitions for Speech Emotion Recognition and Speaker Identification
Extracted from speech.ipynb for deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with improved design"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class AttentionBlock(nn.Module):
    """Squeeze-and-excitation attention block"""
    def __init__(self, channels, reduction=16):
        super(AttentionBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        batch, channels, height, width = x.size()
        se = F.adaptive_avg_pool2d(x, 1).view(batch, channels)
        se = self.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(batch, channels, 1, 1)
        return x * se


class AdvancedEmotionModel(nn.Module):
    """Advanced CNN-LSTM-Attention model for emotion recognition - Target: 95%+ accuracy"""
    
    def __init__(self, input_shape, num_emotions):
        super(AdvancedEmotionModel, self).__init__()
        
        # Initial convolution with stronger regularization
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks with increasing filters
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Attention blocks
        self.attention1 = AttentionBlock(64)
        self.attention2 = AttentionBlock(128)
        self.attention3 = AttentionBlock(256)
        self.attention4 = AttentionBlock(512)
        
        # Deeper LSTM for temporal modeling (3 layers, stronger dropout)
        self.lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True, num_layers=3, dropout=0.6)
        
        # Global average pooling + FC layers
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        
        # Dense layers with batch norm and strong regularization
        self.fc1 = nn.Linear(512 + 512, 1024)  # 512 from CNN + 512 from LSTM, expanded
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout_fc1 = nn.Dropout(0.6)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.dropout_fc2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn_fc3 = nn.BatchNorm1d(256)
        self.dropout_fc3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(256, num_emotions)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = self.initial_conv(x)
        
        # Residual layers with attention
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        x = self.layer4(x)
        x = self.attention4(x)
        
        # CNN path: Global average pooling
        cnn_features = self.gap(x).view(x.size(0), -1)  # (batch, 512)
        
        # LSTM path: Reshape for LSTM
        batch_size = x.size(0)
        x_lstm = x.view(batch_size, x.size(1), -1)  # (batch, 512, H*W)
        x_lstm = x_lstm.permute(0, 2, 1)  # (batch, H*W, 512)
        lstm_out, (h_n, c_n) = self.lstm(x_lstm)
        lstm_features = lstm_out[:, -1, :]  # Take last output (batch, 512)
        
        # Concatenate CNN and LSTM features
        x = torch.cat([cnn_features, lstm_features], dim=1)  # (batch, 1024)
        
        # Dense layers with progressive reduction
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout_fc3(x)
        
        x = self.fc4(x)
        
        return x


class AdvancedSpeakerModel(nn.Module):
    """Advanced CNN model for speaker identification - optimized for high accuracy"""
    
    def __init__(self, input_shape, num_speakers):
        super(AdvancedSpeakerModel, self).__init__()
        
        # Initial convolution with stronger regularization
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense residual blocks for better feature learning
        self.layer1 = self._make_layer(128, 128, 2, stride=1)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        self.layer4 = self._make_layer(512, 512, 2, stride=2)
        
        # Attention blocks for feature refinement
        self.attention1 = AttentionBlock(128)
        self.attention2 = AttentionBlock(256)
        self.attention3 = AttentionBlock(512)
        self.attention4 = AttentionBlock(512)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Dense layers with large bottleneck for speaker-specific features
        self.fc1 = nn.Linear(512, 2048)  # Expanded bottleneck
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)
        self.dropout_fc2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(1024, 512)
        self.bn_fc3 = nn.BatchNorm1d(512)
        self.dropout_fc3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(512, 256)
        self.bn_fc4 = nn.BatchNorm1d(256)
        self.dropout_fc4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(256, num_speakers)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = self.initial_conv(x)
        
        # Residual layers with attention
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        x = self.layer4(x)
        x = self.attention4(x)
        
        # Global average pooling
        x = self.gap(x).view(x.size(0), -1)
        
        # Dense layers with large bottleneck
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout_fc3(x)
        
        x = F.relu(self.bn_fc4(self.fc4(x)))
        x = self.dropout_fc4(x)
        
        x = self.fc5(x)
        
        return x
