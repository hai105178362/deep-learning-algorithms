import torch

# Consider an input vocabulary of 6: ' ', A, C, E, H, T.
input_vocabulary = [' ', 'A', 'C', 'E', 'H', 'T']

# We want to map to an output vocabulary of 8: '-', aa, ch, d, ei, k, t, uh.
output_vocabulary = ['-', 'aa', 'ch', 'd', 'ei', 'k', 't', 'uh']

# Here are two observations with equivalent phrases represented in their respective vocabulary
data = [('TEACH THE CAT', 't-ei-t-ch-d-uh-k-aa-t'),
        ('CATCH THE CHEAT', 'k-aa-t-ch-d-uh-t-ch-ei-t')]

X = [torch.LongTensor([input_vocabulary.index(iv) for iv in spell]) for spell, _ in data]
# print("Shapes in prepared X:\n\t", [x.shape for x in X])

Y = [torch.LongTensor([output_vocabulary.index(ov) for ov in speak.split("-")]) for _, speak in data]
# print("\nShapes in prepared Y:\n\t", [y.shape for y in Y])

X_lens = torch.LongTensor([len(seq) for seq in X])
# print("\nShape of prepared X lengths:\n\t", X_lens)

Y_lens = torch.LongTensor([len(seq) for seq in Y])
# print("\nShape of prepared Y lengths:\n\t", Y_lens)

X = torch.nn.utils.rnn.pad_sequence(X)
# print("\nShape of padded X:\n\t", X.shape)

Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True)
# print("\nShape of padded Y:\n\t", Y.shape)


class Model(torch.nn.Module):
    def __init__(self, in_vocab, out_vocab, embed_size, hidden_size):
        super(Model, self).__init__()
        self.embed = torch.nn.Embedding(in_vocab, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.output = torch.nn.Linear(hidden_size * 2, out_vocab)

    def forward(self, X, lengths):
        print("\nShape of Input X:\n\t", X.shape)

        X = self.embed(X)
        print("\nShape of X Embedded: \n\t", X.shape)
        # print(X)
        exit()
        packed_X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, enforce_sorted=False)
        print("\nShapes in packed embedding: \n\t", [px.shape for px in packed_X])

        packed_out = self.lstm(packed_X)[0]
        print("\nShapes in LSTM Output: \n\t", [po.shape for po in packed_out])

        out, out_lens = torch.nn.utils.rnn.pad_packed_sequence(packed_out)
        print("\nShape of packed LSTM Output: \n\t", out.shape)
        print("\nShape of packed LSTM Output lengths: \n\t", out_lens.shape)

        out = self.output(out).log_softmax(2)
        print("\nShape of post-softmax of packed LSTM Output:\n\t", out.shape)
        return out, out_lens


Model(len(input_vocabulary), len(output_vocabulary), 4, 8)(X, X_lens)

print("\nfin.")