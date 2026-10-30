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


def test_on_demand_stats_match_main_float32_reductions(tmp_path):
    from tmg.data.loader import TrajectoryStore, compute_stats
    from types import SimpleNamespace
    rng=np.random.default_rng(41)
    arrays=[rng.integers(0,256,(n,12,13,3),dtype=np.uint8) for n in (31,19,43)]
    path=tmp_path/'sampled.hdf5'
    with h5py.File(path,'w') as f:
        for i,a in enumerate(arrays): f.create_dataset(str(i),data=a)
    owner=[str(path),None]
    vectors=[rng.normal(size=(len(a),4)).astype(np.float32) for a in arrays]
    store=SimpleNamespace(images={'camera':[H5Frames(owner,str(i),a.shape) for i,a in enumerate(arrays)]},
                          cameras=['camera'],image_shape=(12,13,3),state=vectors,actions=vectors,env_state=None)
    try:
        for limit in (100,37):
            actual=compute_stats(store,max_image_frames=limit,seed=7)['observation.images.camera']
            frames=np.concatenate(arrays)
            if len(frames)>limit: frames=frames[np.sort(np.random.default_rng(7).choice(len(frames),limit,replace=False))]
            x=frames.astype(np.float32)/255.0
            for key in ('min','max','mean','std'):
                expected=getattr(x,key)(axis=(0,1,2)).reshape(3,1,1)
                assert np.array_equal(actual[key],expected), key
    finally:
        if owner[1] is not None:owner[1].close()
