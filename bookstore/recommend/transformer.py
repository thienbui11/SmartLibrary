# bookapp/recommend/transformer.py
import torch.nn as nn

class RecTransformer(nn.Module):
    def __init__(self, embed_dim=384, n_books=1000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Sửa tại đây: Thêm batch_first=True và phù hợp với TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=4,
            batch_first=True  # Quan trọng
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        
        self.book_head = nn.Linear(embed_dim, n_books)

    def forward(self, x):
        # Input x phải có dạng (batch_size, seq_len, embed_dim)
        x = self.transformer(x)
        x = x[:, -1, :]  # Lấy phần tử cuối của sequence
        return self.book_head(x)  # Output: (batch_size, n_books)