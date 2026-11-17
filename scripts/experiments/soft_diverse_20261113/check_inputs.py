"""Verify that adding annotations preserved every original dataset and episode attribute."""
import json
import re
from pathlib import Path

import h5py
import numpy as np

base = Path(__file__).resolve().parent
all_operators = set()
report = {}
for task in ('rope','towel'):
    comparisons = []
    with h5py.File(base/f'{task}.hdf5') as assembled:
        for i in range(1,11):
            label=f'{task}{i:02d}'
            folder=base/label
            log=(folder/'agent.log').read_text()
            operator=re.search(r'^session id: (\S+)',log,re.M).group(1)
            assert re.search(r'^model: gpt-6-astra$',log,re.M)
            assert operator not in all_operators
            all_operators.add(operator)
            selected=json.loads((folder/'input.json').read_text())
            decision=json.loads((folder/'boundary.json').read_text())
            g=assembled[f'data/demo_{i-1}']
            with h5py.File(selected['file']) as source:
                orig,=source['data'].values()
                for key,value in orig.attrs.items():
                    np.testing.assert_array_equal(g.attrs[key],value)
                checked=[]
                def compare(key, obj):
                    if isinstance(obj,h5py.Dataset):
                        new=g[key]; assert new.dtype==obj.dtype and new.shape==obj.shape
                        for start in range(0,len(obj),24):
                            np.testing.assert_array_equal(new[start:start+24],obj[start:start+24])
                        checked.append(key)
                orig.visititems(compare)
            comparisons.append(dict(scene=label,operator=operator,raw_datasets_equal=len(checked),boundary=decision['boundary'],confidence=decision['confidence']))
    report[task]=comparisons
assert len(all_operators)==20
(base/'preservation_verified.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
