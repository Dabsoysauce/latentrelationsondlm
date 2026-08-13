import torch

from dlmrel.experiments.ablation import paired_ablation_rows
from dlmrel.experiments.dla import direct_logit_attribution
from dlmrel.models.fake import FakeAdapter


def test_fake_shapes_dla_and_ablation_isolation():
    adapter = FakeAdapter()
    ids = torch.tensor([[1, 2, 3, 4]])
    output = adapter.forward(ids, timestep=2)
    assert output.logits.shape == (1, 4, 32)
    assert len(output.attentions) == 2
    unembed = torch.ones(12, 32)
    score = direct_logit_attribution(
        output.head_outputs[0, 0, 0], unembed, query_position=1, output_token_id=2
    )
    assert score == 1.0
    rows = paired_ablation_rows(
        adapter,
        [
            {
                "sentence_id": "s",
                "instance_id": "i",
                "input_ids": ids,
                "target_position": 1,
                "target_token": 2,
            }
        ],
        layer=0,
        head=0,
    )
    assert len(rows) == 1
    assert rows.iloc[0].logit_delta != 0
