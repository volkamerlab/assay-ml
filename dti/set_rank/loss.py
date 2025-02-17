import torch
from torch.nn.functional import cross_entropy
from torch import logsumexp, Tensor, softmax


def log1mexp(x: Tensor, dim: int):
    m = x.max(dim, keepdim=True).values
    shifted_exp_x = torch.exp(x - m)
    return m + torch.log(shifted_exp_x.sum(dim, keepdim=True) - shifted_exp_x)


def ranking_cross_entropy_with_logits2(s: Tensor, y: Tensor, dim: int = 0):
    """
    https://arxiv.org/pdf/1912.05891 "model training" section
    Args:
        s (Tensor): unnormalized ranking scores
        y (Tensor): target relevance scores
        dim (int, optional): Dimension along which target ranking scores should be softmaxes. Defaults to 0.
    """
    y = y.view(s.size())
    py = softmax(y, dim=dim)
    lse_s = logsumexp(s, dim=dim)
    positive = py * (s - lse_s)
    negative = (1 - py) * (log1mexp(s, dim=dim) - lse_s)
    return -(positive + negative).mean()


def ranking_cross_entropy_with_logits3(s: Tensor, y: Tensor, dim: int = 0):
    """
    https://arxiv.org/pdf/1912.05891 "model training" section
    Args:
        s (Tensor): unnormalized ranking scores
        y (Tensor): target relevance scores
        dim (int, optional): Dimension along which target ranking scores should be softmaxes. Defaults to 0.
    """
    ps = s.softmax(dim=dim)
    py = y.softmax(dim=dim)
    return -(py * ps.log() + (1 - py) * (1 - ps).log()).mean()


def ranking_cross_entropy_with_logits(s: Tensor, y: Tensor, dim: int = 0):
    """
    https://arxiv.org/pdf/1912.05891 "model training" section

    Only supports batch size 1.

    Args:
        s (Tensor): unnormalized ranking scores
        y (Tensor): target relevance scores
        dim (int, optional): Dimension along which target ranking scores should be softmaxes. Defaults to 0.
    """
    y = y.view(s.size())
    py = softmax(y, dim=dim)
    ce = cross_entropy(s.view(1, -1), py.view(1, -1), reduction="none")
    return ce


if __name__ == "__main__":
    import torch

    s1 = torch.tensor([1, 2, 1, 3, 5]).view(-1, 1).float()
    s1 = ((s1 - s1.mean()) / s1.std()).round(decimals=2)
    s2 = torch.tensor([2, 3, 1, 2, 4]).view(-1, 1).float()
    s2 = ((s2 - s2.mean()) / s2.std()).round(decimals=2)
    y = torch.tensor([0, 1, 2, 3, 4]).view(-1, 1).float()

    l11 = ranking_cross_entropy_with_logits(s1, y)
    print(f"loss1: {l11} \n\t s: {s1.flatten()} \n\t y: {y.flatten()}")

    l12 = ranking_cross_entropy_with_logits(s2, y)
    print(f"loss1: {l12} \n\t s: {s2.flatten()} \n\t y: {y.flatten()}")

    l21 = ranking_cross_entropy_with_logits2(s1, y)
    print(f"loss2: {l21} \n\t s: {s1.flatten()} \n\t y: {y.flatten()}")

    l22 = ranking_cross_entropy_with_logits2(s2, y)
    print(f"loss2: {l22} \n\t s: {s2.flatten()} \n\t y: {y.flatten()}")

    lopt = ranking_cross_entropy_with_logits2(y, y)
    print(f"loss2: {lopt} \n\t s: {y.flatten()} \n\t y: {y.flatten()}")

    y = torch.tensor([1, 3, 3, 1, 1, 5, 10]).view(-1, 1).float()
    s = torch.randn(y.size(0), 1, requires_grad=True)
    print(s.is_leaf)
    print(s.flatten())
    opt = torch.optim.Adam([s], lr=0.01)
    for i in range(100):
        loss = ranking_cross_entropy_with_logits(s, y)
        if torch.isinf(loss):
            print("inf")
            loss = ranking_cross_entropy_with_logits(s, y)
            break
        if torch.isnan(loss):
            print("nan")
            loss = ranking_cross_entropy_with_logits(s, y)
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
    print(s.flatten())
    print(y.flatten())
