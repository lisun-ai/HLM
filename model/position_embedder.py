# This piece of code is from https://github.com/microsoft/DeBERTa

import torch


class PositionEmbedder(torch.nn.Module):
    def __init__(self, embedding_dim, max_num_word=1024, max_char_per_word=128):
        super(PositionEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding_word = torch.nn.Embedding(max_num_word, embedding_dim, padding_idx=None)
        self.embedding_char = torch.nn.Embedding(max_char_per_word, embedding_dim, padding_idx=None)

    def forward(self, input_embeddings: torch.Tensor):
        # positional embeddings will broadcast since they're just missing batch dim
        # input_embeddings: [max_char_len, max_num_word, batch_size, hidden_size]
        # positions_word: [max_num_word, hidden_size//2]
        positions_word = torch.arange(input_embeddings.shape[1], device=input_embeddings.device)
        positions_word = self.embedding_word(positions_word)
        input_embeddings = input_embeddings + positions_word[None, :, None, :]
        # positions_char: [max_char_len, hidden_size//2]
        positions_char = torch.arange(input_embeddings.shape[0], device=input_embeddings.device)
        positions_char = self.embedding_char(positions_char)
        input_embeddings = input_embeddings + positions_char[:, None, None, :]
        return input_embeddings