import torch.nn as nn


class CRNN(nn.Module):
    """
    A CRNN (Convolutional Recurrent Neural Network) for Optical Character Recognition (OCR).
    This architecture combines convolutional layers for feature extraction and LSTM layers for sequence modeling.

    Args:
        imgH (int): Height of the input image (default=32).
        num_channels (int): Number of input channels (default=1, grayscale images).
        num_classes (int): Number of output classes (default=66, includes all characters + blank for CTC).
        hidden_units (int): Number of hidden units in the LSTM layers (default=256).
    """
    def __init__(self, imgH=32, num_channels=1, num_classes=66, hidden_units=256):
        super(CRNN, self).__init__()

        # Convolutional layers for feature extraction
        self.cnn = nn.Sequential(
            # First convolution block
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),  # Conv1: 64 feature maps, 3x3 kernel
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Downsample by a factor of 2

            # Second convolution block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Conv2: 128 feature maps, 3x3 kernel
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Downsample by a factor of 2

            # Third convolution block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Conv3: 256 feature maps, 3x3 kernel
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Conv4: 256 feature maps, 3x3 kernel
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Downsample width by 2 (height remains the same)

            # Fourth convolution block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Conv5: 512 feature maps, 3x3 kernel
            nn.BatchNorm2d(512),  # Batch normalization to stabilize training
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Conv6: 512 feature maps, 3x3 kernel
            nn.BatchNorm2d(512),  # Batch normalization
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Downsample width by 2 (height remains the same)

            # Fifth convolution block
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),  # Conv7: 512 feature maps, 2x2 kernel
        )

        # Recurrent layers for sequence modeling
        self.rnn1 = nn.LSTM(512, hidden_units, bidirectional=True, batch_first=True)  # First bidirectional LSTM
        self.rnn2 = nn.LSTM(hidden_units * 2, hidden_units, bidirectional=True, batch_first=True)  # Second LSTM

        # Fully connected layer to project LSTM output to class probabilities
        self.fc = nn.Linear(hidden_units * 2, num_classes)

    def forward(self, x):
        """
        Forward pass of the CRNN model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Log-probabilities over classes of shape (batch_size, seq_len, num_classes).
        """
        # CNN feature extraction
        conv_out = self.cnn(x)  # Shape: (batch_size, 512, height / 16, width / 16)

        # Ensure the height dimension is 1 (required for sequence modeling)
        b, c, h, w = conv_out.size()
        assert h == 1, "Height of convolutional output must be 1. Check input dimensions."

        # Remove the height dimension and permute for LSTM
        conv_out = conv_out.squeeze(2)  # Shape: (batch_size, 512, width / 16)
        conv_out = conv_out.permute(0, 2, 1)  # Shape: (batch_size, width / 16, 512)

        # Sequence modeling with LSTM
        rnn_out, _ = self.rnn1(conv_out)  # First LSTM layer
        rnn_out, _ = self.rnn2(rnn_out)  # Second LSTM layer

        # Fully connected layer for classification
        output = self.fc(rnn_out)  # Shape: (batch_size, seq_len, num_classes)

        # Apply log-softmax along the class dimension
        return output.log_softmax(2)

    
