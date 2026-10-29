import hashlib
import h5py
import numpy as np
from tmg.generation.generators.isaaclab.sources import compatible_source


def test_object_alias_preserves_source_and_pose_arrays(tmp_path):
    source=tmp_path/'source.hdf5'
    poses=np.arange(32).reshape(2,4,4)
    with h5py.File(source,'w') as f:
        f.create_dataset('data/demo_0/obs/datagen_info/object_pose/old',data=poses)
        f.create_dataset('data/demo_0/actions',data=np.ones((2,36)))
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    task=dict(source=str(source),source_object_aliases={'old':'new'})
    prepared=compatible_source(task)
    assert prepared!=source
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    with h5py.File(prepared,'r') as f:
        group=f['data/demo_0/obs/datagen_info/object_pose']
        assert group['old'].id==group['new'].id
        assert np.array_equal(group['new'][:],poses)
        assert np.array_equal(f['data/demo_0/actions'][:],np.ones((2,36)))
    assert compatible_source(task)==prepared
