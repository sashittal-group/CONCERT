from src.coord_diff_top_batches import (
    LINEAR_TOP_FIFTY_BATCHES,
    LINEAR_TOP_TWENTY_BATCHES,
    STAR3_TOP_FIFTY_BATCHES,
    STAR3_TOP_TWENTY_BATCHES,
)


def test_top_fifty_selections_extend_top_twenty_with_distinct_settings():
    for top_twenty, top_fifty in (
        (STAR3_TOP_TWENTY_BATCHES, STAR3_TOP_FIFTY_BATCHES),
        (LINEAR_TOP_TWENTY_BATCHES, LINEAR_TOP_FIFTY_BATCHES),
    ):
        assert len(top_twenty) == 20
        assert len(top_fifty) == 50
        assert top_fifty[:20] == top_twenty
        assert len({setting for setting, _ in top_fifty}) == 50
