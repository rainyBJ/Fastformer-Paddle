from fast_transformer_pd import  FastTransformer
from paddle import nn
import paddle

# model
model = FastTransformer(
    num_tokens = 77053,
    dim = 512,
    depth = 2,
    max_seq_len = 512,
    absolute_pos_emb = True,
    dropout=0.2 # default uses relative positional encoding, but if that isn't working, then turn on absolute positional embedding by setting this to True
)

class Model(nn.Layer):

    def __init__(self, ):
        super(Model, self).__init__()
        self.dense_linear = nn.Linear(512, 5)
        self.fastformer_model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, targets):
        mask = paddle.to_tensor(input_ids).astype("bool")
        text_vec = self.fastformer_model(input_ids, mask)
        score = self.dense_linear(text_vec)
        loss = self.criterion(score, targets)
        return loss, score