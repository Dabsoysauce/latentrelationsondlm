import pytest

from dlmrel.experiments.native_timing import timing_rows


def test_native_timing_is_explicitly_separate():
    output = timing_rows(
        [
            {
                "prompt": "The chef",
                "generated_token": "cooked",
                "native_step": 3,
                "found_step": 2,
                "reveal_step": 4,
                "seed": 42,
                "sampler_settings": {"steps": 8},
            }
        ]
    )
    assert output.iloc[0].trajectory == "native_generated"


def test_native_timing_requires_found_and_reveal_steps():
    with pytest.raises(ValueError, match="missing"):
        timing_rows([{"prompt": "x"}])
