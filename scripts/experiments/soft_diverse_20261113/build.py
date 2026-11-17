"""Assemble 10 pre-action annotated demonstrations, preserving all recorded datasets."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

BASE = Path(__file__).resolve().parent
NODES = 'object_nodal_position/object'

def pre_nodes(episode, task):
    if task == 'rope':
        return episode['obs/object_nodal_pos'][:]
    key = 'deformable/object/nodal_position'
    return np.concatenate([episode['initial_state/' + key][:], episode['states/' + key][:-1]])

def transforms(episode):
    obs = episode['obs']
    xyz = obs['robot0_eef_pos'][:]
    quat = obs['robot0_eef_quat'][:]
    act = episode['actions'][:]
    pose = np.tile(np.eye(4,dtype='f4'), (len(act),1,1))
    pose[:,:3,3] = xyz
    pose[:,:3,:3] = Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
    target = pose.copy()
    target[:,:3,3] += act[:,:3]
    target[:,:3,:3] = Rotation.from_rotvec(act[:,3:6]).as_matrix() @ pose[:,:3,:3]
    return pose,target

def annotate(g,task,idx,manual=None):
    pose,target = transforms(g)
    nodes = pre_nodes(g,task)
    if task == 'rope':
        dist = np.linalg.norm(nodes-pose[:,None,:3,3],axis=2).min(1)
        joints = g['obs/robot0_gripper_qpos'][:] * [1,-1]
        flags = (dist < .015) & (np.abs(joints-.04)>.005).all(1)
    else:
        nodes = nodes[:,idx]
        flags = (np.arange(len(nodes)) >= manual)[None,:]
    key = 'grasp' if task=='rope' else 'grasp_1'
    return {'eef_pose/robot0':pose, 'target_eef_pose/robot0':target,
            NODES:nodes,'subtask_term_signals/'+key:flags}

def build(task):
    records = []
    with h5py.File(BASE/'reference'/f'{task}.hdf5') as ref:
        idx = None
        ref_data = ref['data']
        for g in ref_data.values():
            official = g['obs/datagen_info']
            if task=='towel':
                full = pre_nodes(g,task)
                # Identify the released CUDA seed-42 vertex permutation by exact coordinate
                # correspondence and validate every selected coordinate at every timestep.
                ids = cKDTree(full[10]).query(official[NODES][10])[1]
                assert len(set(ids))==100 and (idx is None or np.array_equal(ids,idx))
                idx = ids
                manual = int(np.flatnonzero(official['subtask_term_signals/grasp_1'][:].ravel())[0])
            else: manual = None
            for key,arr in annotate(g,task,idx,manual).items():
                orig=official[key][:]
                assert arr.shape==orig.shape and arr.dtype==orig.dtype
                if 'eef_pose' in key:
                    np.testing.assert_allclose(arr,orig,atol=1e-6,rtol=0)
                else: np.testing.assert_array_equal(arr,orig)
        tmp=BASE/(task+'.partial.hdf5')
        with h5py.File(tmp,'w') as dest:
            data=dest.create_group('data')
            for k,v in ref_data.attrs.items(): data.attrs[k]=v
            total=0
            for i in range(1,11):
                folder=BASE/f'{task}{i:02d}'
                meta=json.loads((folder/'input.json').read_text())
                decision=json.loads((folder/'boundary.json').read_text())
                assert decision['source_sha256']==meta['sha256']
                with open(meta['file'],'rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==meta['sha256']
                with h5py.File(meta['file']) as f:
                    g,=f['data'].values()
                    assert g.attrs['success'] and 'datagen_info' not in g['obs']
                    boundary=int(decision['boundary'])
                    arrs=annotate(g,task,idx,boundary)
                    flags=next(v for k,v in arrs.items() if 'signals' in k).ravel()
                    assert int(np.flatnonzero(flags)[0])==boundary and 0<boundary<len(flags)-2
                    name=f'demo_{i-1}'
                    f.copy(g,data,name=name)
                    info=data[name]['obs'].create_group('datagen_info')
                    for key,arr in arrs.items():
                        assert np.isfinite(arr).all()
                        info.create_dataset(key,data=arr,compression='gzip',compression_opts=1)
                    total+=len(flags)
                    records.append(dict(scene=meta['scene'],demo=name,source_sha256=meta['sha256'],boundary=boundary,steps=len(flags),agent_evidence=decision['evidence']))
            data.attrs['total']=total
        tmp.replace(BASE/f'{task}.hdf5')
        (BASE/f'{task}_annotations.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
if __name__=='__main__':
    import sys
    build(sys.argv[1])
