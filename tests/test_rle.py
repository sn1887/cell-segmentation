import numpy as np

from cellseg_challenge.utils import rle_decode, rle_encode


def test_rle_roundtrip_row_major():
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[0, :2] = 1
    mask[2, 3:] = 1

    encoded = rle_encode(mask)

    assert encoded == "1 2 14 2"
    np.testing.assert_array_equal(rle_decode(encoded, mask.shape), mask)

