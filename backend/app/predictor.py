import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM with linear projection"""

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size,
                           bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(x)
        output = self.linear(recurrent)
        return output


class CRNN(nn.Module):
    """CRNN model for OCR"""

    def __init__(self, img_height, num_channels, num_classes,
                 hidden_size=256, num_lstm_layers=2, dropout=0.1):
        super(CRNN, self).__init__()

        assert img_height % 16 == 0, "img_height must be divisible by 16"

        self.img_height = img_height
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

        # CNN Feature Extraction
        self.cnn = self._build_cnn()
        self.lstm_input_size = 512

        # RNN Sequence Modeling
        self.rnn_layers = nn.ModuleList()
        for i in range(num_lstm_layers):
            input_size = self.lstm_input_size if i == 0 else hidden_size

            if i < num_lstm_layers - 1 and dropout > 0:
                self.rnn_layers.append(
                    nn.Sequential(
                        BidirectionalLSTM(
                            input_size, hidden_size, hidden_size),
                        nn.Dropout(dropout)
                    )
                )
            else:
                self.rnn_layers.append(
                    BidirectionalLSTM(input_size, hidden_size, hidden_size)
                )

        # Output Layer
        self.output = nn.Linear(hidden_size, num_classes)
        self._initialize_weights()

    def _build_cnn(self):
        """Build standard CNN backbone"""
        cnn = nn.Sequential(
            # Layer 1: (batch, 1, 64, W) -> (batch, 64, 64, W)
            nn.Conv2d(self.num_channels, 64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 64, 32, W/2)

            # Layer 2: (batch, 64, 32, W/2) -> (batch, 128, 32, W/2)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 128, 16, W/4)

            # Layer 3: (batch, 128, 16, W/4) -> (batch, 256, 16, W/4)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Layer 4: (batch, 256, 16, W/4) -> (batch, 256, 16, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # (batch, 256, 8, W/4)
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Layer 5: (batch, 256, 8, W/4) -> (batch, 512, 8, W/4)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Layer 6: (batch, 512, 8, W/4) -> (batch, 512, 8, W/4)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # (batch, 512, 4, W/4)
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Layer 7: (batch, 512, 4, W/4) -> (batch, 512, 2, W/4)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # (batch, 512, 2, W/4)
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Layer 8: (batch, 512, 2, W/4) -> (batch, 512, 1, W/4)
            nn.Conv2d(512, 512, kernel_size=(2, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # (batch, 512, 1, W/4)
        )
        return cnn

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass"""
        conv = self.cnn(x)
        batch, channels, height, width = conv.size()

        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1).contiguous()

        output = conv
        for rnn_layer in self.rnn_layers:
            output = rnn_layer(output)

        output = self.output(output)
        output = output.permute(1, 0, 2)
        output = F.log_softmax(output, dim=2)

        return output

    def get_total_parameters(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


class CTCLabelConverter:
    """Convert between text labels and CTC indices"""

    def __init__(self, charset):
        self.charset = charset
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(charset)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(charset)}
        self.blank_idx = 0
        self.num_classes = len(charset) + 1

    def decode(self, indices):
        """Decode indices to text string"""
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()

        chars = []
        for idx in indices:
            if idx == self.blank_idx:
                continue
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])

        return ''.join(chars)

    def ctc_greedy_decode(self, log_probs):
        """Greedy CTC decoding"""
        _, max_indices = torch.max(log_probs, dim=2)
        max_indices = max_indices.transpose(0, 1)

        decoded_texts = []
        for indices in max_indices:
            indices = indices.cpu().numpy()
            decoded_indices = []
            prev_idx = None

            for idx in indices:
                if idx != self.blank_idx and idx != prev_idx:
                    decoded_indices.append(idx)
                prev_idx = idx

            text = self.decode(decoded_indices)
            decoded_texts.append(text)

        return decoded_texts


class OCRPredictor:
    """OCR Predictor for inference"""

    def __init__(self, model_path, charset_path, device=None):
        """
        Initialition of OCR Predictor yeha hudai xa 

        Args:
            model_path: drive bata download gareko model ko path
            charset_path: charset.txt ko path
            device: Device to use (auto-detect if None)
        """
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Initializing OCR Predictor on {self.device}...")

        # Load charset
        with open(charset_path, 'r', encoding='utf-8') as f:
            charset = f.read().strip()

        self.converter = CTCLabelConverter(charset)
        num_classes = self.converter.num_classes

        print(f"✓ Charset loaded: {len(charset)} characters")

        # Create model
        self.model = CRNN(
            img_height=64,
            num_channels=1,
            num_classes=num_classes,
            hidden_size=128,
            num_lstm_layers=2,
            dropout=0.1
        ).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ Model loaded successfully")
        print(f"✓ Model parameters: {self.model.get_total_parameters():,}")

    def preprocess_for_model(self, preprocessed_image_path):
        """
        Convert preprocessed image to model input tensor

        Args:
            preprocessed_image_path: Path to preprocessed image (already cleaned)

        Returns:
            Image tensor ready for model
        """
        # Load preprocessed image (already grayscale, thresholded)
        image = Image.open(preprocessed_image_path).convert('L')

        # Convert to tensor
        image_tensor = transforms.ToTensor()(image)

        # Normalize: invert and scale to [-1, 1]
        image_tensor = 1.0 - image_tensor  # Invert
        image_tensor = (image_tensor - 0.5) / 0.5  # Normalize to [-1, 1]

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def predict(self, preprocessed_image_path):
        """
        Predict text from preprocessed image

        Args:
            preprocessed_image_path: Path to preprocessed image

        Returns:
            tuple: (predicted_text, confidence)
        """
        # Convert to model input
        image_tensor = self.preprocess_for_model(
            preprocessed_image_path).to(self.device)

        # Inference
        with torch.no_grad():
            log_probs = self.model(image_tensor)

            # Decode
            predictions = self.converter.ctc_greedy_decode(log_probs)
            prediction = predictions[0]

            # Calculate confidence
            probs = torch.exp(log_probs)
            max_probs, _ = torch.max(probs, dim=2)
            confidence = max_probs.mean().item()

        return prediction, confidence
