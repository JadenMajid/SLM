from tokenizer import BigramDataset, import_southpark
import numpy as np
from mathutils import softmax
import pickle

import numpy as np

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

class BigramModel:
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset.X = np.array(dataset.X, dtype=np.int32)

        # compute the correct vocab size
        vocab_size = int(np.max(self.dataset.X)) + 1
        self.dataset.d = vocab_size

        # now allocate embeddings
        self.embedding = (0.5 - np.random.rand(vocab_size, 8)).astype(np.float32) / np.sqrt(vocab_size)
        self.l1 = (0.5 - np.random.rand(8, 8)).astype(np.float32) / np.sqrt(vocab_size)
        self.l2 = (0.5 - np.random.rand(8, vocab_size)).astype(np.float32) / np.sqrt(vocab_size)

        self.learn_rate = 0.01
        self.max_iterations = 3
        self.batch_size = 512


    def predict_one(self, idx):
        h = np.maximum(0, self.embedding[idx] @ self.l1)
        out = h @ self.l2
        return softmax(out)

    def loss(self, y_hat, target):
        return -np.log(y_hat[target] + 1e-9)

    def fit(self):
        # filter invalid indices
        valid_mask = self.dataset.X < self.embedding.shape[0]
        self.dataset.X = self.dataset.X[valid_mask]

        X = self.dataset.X
        n = len(X)

        for epoch in range(self.max_iterations):
            total_loss = 0.0
            perm = np.random.permutation(n)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch = perm[start:end]

                grad_l2 = np.zeros_like(self.l2)
                grad_l1 = np.zeros_like(self.l1)
                grad_embed = np.zeros_like(self.embedding)

                for j in batch:
                    y_hat = self.predict_one(j)
                    loss = self.loss(y_hat, j)
                    total_loss += loss

                    diff = y_hat
                    diff[j] -= 1  # gradient wrt output layer

                    h = np.maximum(0, self.embedding[j] @ self.l1)

                    grad_l2 += np.outer(h, diff)
                    dh = diff @ self.l2.T
                    dh[h <= 0] = 0
                    grad_l1 += np.outer(self.embedding[j], dh)
                    grad_embed[j] += dh @ self.l1.T

                bs = len(batch)
                self.l2 -= self.learn_rate * grad_l2 / bs
                self.l1 -= self.learn_rate * grad_l1 / bs
                self.embedding -= self.learn_rate * grad_embed / bs

            print(f"epoch {epoch}: loss={total_loss / n:.6f}")



        




if __name__ == "__main__":
    
    with open('data/bigramdataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    model = BigramModel(dataset)
    
    model.fit()

    with open('data/model.pkl', 'wb') as f:
        pickle.dump(model, f)