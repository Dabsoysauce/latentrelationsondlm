import torch

from dlmrel.models.fake import FakeAdapter


def test_fake_shapes_are_deterministic():
    adapter = FakeAdapter()
    ids = torch.tensor([[1, 2, 3, 4]])
    output = adapter.forward(ids, timestep=2)
    repeated = adapter.forward(ids, timestep=2)
    assert output.logits.shape == (1, 4, 32)
    assert len(output.attentions) == 2
    assert torch.equal(output.logits, repeated.logits)
    assert all(
        torch.equal(left, right)
        for left, right in zip(output.attentions, repeated.attentions, strict=True)
    )
