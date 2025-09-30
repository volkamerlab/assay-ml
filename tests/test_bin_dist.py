import torch

from dti.model import BinDistribution


def test_edges():
    dist = BinDistribution(5)
    # (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    edges = dist._construct_edges()
    print("edges", edges)
    assert edges.shape == (6,)
    assert edges.allclose(torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    return dist, edges


def test_center():
    dist, edges = test_edges()
    centers = dist.bucket_centers()
    print("center", centers)


def test_mean():
    dist, _ = test_edges()
    logits = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32).unsqueeze(0)
    mu = dist.mean(logits)
    print("mean", mu)
    assert torch.isclose(mu, torch.tensor(0.5))


def test_icdf1():
    dist, edges = test_edges()
    logits = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32).unsqueeze(0)
    print("p", logits.softmax(-1))
    q = torch.tensor([0.25, 0.5, 0.75])
    xq = dist.icdf(logits, q)
    assert torch.allclose(xq, q)
    print("xq", xq)


def test_icdf2():
    dist, edges = test_edges()
    logits = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32).unsqueeze(0)
    print("p", logits.softmax(-1))
    q = torch.tensor([0.01, 1.0])
    xq = dist.icdf(logits, q)
    assert torch.allclose(xq, q)
    print("xq", xq)


def test_median():
    dist = BinDistribution(4)
    logits = torch.tensor([4, 2, 1, 1]).log().unsqueeze(0)
    median = dist.median(logits)
    print("median", median)
    assert median.item() == 0.25

    logits = torch.tensor([1, 1, 2, 4]).log().unsqueeze(0)
    assert dist.median(logits).item() == 0.75


if __name__ == "__main__":
    test_edges()
    test_mean()
    test_center()
    test_icdf1()
    test_icdf2()
    test_median()
