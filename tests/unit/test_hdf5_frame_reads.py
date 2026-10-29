import h5py
import numpy as np
import pytest
from tmg.data.loader import H5Frames


def test_slice_runs_preserve_order_padding_and_values(tmp_path):
    path=tmp_path/'frames.hdf5'
    expected=np.arange(12*4*5*3,dtype=np.uint8).reshape(12,4,5,3)
    with h5py.File(path,'w') as f: f.create_dataset('images',data=expected,chunks=(1,4,5,3),compression='gzip')
    owner=[str(path),None]
    frames=H5Frames(owner,'images',expected.shape)
    try:
        for indices in [[8,2,3,3,4,10],[0,1,2,3],[-1,0,-1],[]]:
            assert np.array_equal(frames[indices],expected[np.asarray(indices,dtype=int)])
        for indices in [[12],[-13]]:
            with pytest.raises(IndexError): frames[indices]
    finally:
        if owner[1] is not None: owner[1].close()
