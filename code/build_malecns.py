"""
Prepare the MaleCNS connectome for the model.

MaleCNS is one male fly's entire central nervous system -- brain and ventral
nerve cord, imaged together, segmented together. Nothing has to be sewn and
nothing is a chimera: descending neurons arrive with both halves already
attached, and brain and cord share one coordinate frame because they are one
volume.

It also carries its own neurotransmitter predictions, including histamine.
The FlyWire pipeline in this repository needs prepare_neurotransmitters.py to
recover that, because the classifier behind the shipped data emits six
transmitters and cannot name histamine at all; here it is simply a column.

Outputs are written in the format the model already reads:
    data/malecns_completeness.csv
    data/malecns_connectivity.parquet

Source: https://male-cns.janelia.org/download/ (CC-BY, Janelia FlyEM).
Fetch the three inputs first:

    B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
    curl -O $B/body-annotations-male-cns-v1.0-minconf-0.5.feather
    curl -O $B/body-neurotransmitters-male-cns-v1.0.feather
    curl -O $B/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather

Usage:
    python code/build_malecns.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'data/malecns'

ANN = 'body-annotations-male-cns-v1.0-minconf-0.5.feather'
NTS = 'body-neurotransmitters-male-cns-v1.0.feather'
WTS = 'connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather'

# The convention this repository uses throughout: acetylcholine excites, the
# chloride-gated transmitters inhibit. Monoamines act through G-protein-coupled
# receptors and have no fast ionotropic sign, so they take the excitatory
# default that the majority rule in Shiu et al. would also have given them --
# recorded here rather than hidden, because it is a modelling choice.
NT_SIGN = {
    'acetylcholine': 1,
    'gaba': -1,
    'glutamate': -1,
    'histamine': -1,
    'glycine': -1,
    'dopamine': 1,
    'octopamine': 1,
    'serotonin': 1,
    'tyramine': 1,
    'unclear': 1,
}


def build(out_comp, out_conn):
    for f in (ANN, NTS, WTS):
        if not (SRC / f).exists():
            sys.exit(f'{SRC / f} not found; see the module docstring for the '
                     f'download commands')

    ann = pd.read_feather(SRC / ANN)
    traced = ann[ann.status == 'Traced'].drop_duplicates('bodyId')
    ids = np.sort(traced.bodyId.to_numpy(dtype='int64'))
    index_of = pd.Series(np.arange(len(ids)), index=ids)
    print(f'traced neurons: {len(ids)}')

    by_super = traced.superclass.fillna('unknown').value_counts()
    cord = int(by_super.filter(like='vnc').sum())
    print(f'  brain {len(ids) - cord}, ventral nerve cord {cord}, '
          f'descending {int((traced.superclass == "descending_neuron").sum())}, '
          f'motor {int(traced.superclass.isin(["vnc_motor", "cb_motor"]).sum())}')

    nt = pd.read_feather(SRC / NTS)[['body', 'consensus_nt', 'ground_truth']]
    nt = nt.drop_duplicates('body').set_index('body')
    cls = nt.consensus_nt.reindex(ids).fillna('unclear').str.lower()
    unknown = set(cls.unique()) - set(NT_SIGN)
    if unknown:
        sys.exit(f'transmitter classes with no sign: {sorted(unknown)}')
    sign = cls.map(NT_SIGN).astype('int64')

    missing = int(nt.consensus_nt.reindex(ids).isna().sum())
    unclear = int((cls == 'unclear').sum())
    print(f'transmitters: {int((sign > 0).sum())} excitatory, '
          f'{int((sign < 0).sum())} inhibitory')
    print(f'  {int(nt.ground_truth.reindex(ids).notna().sum())} with a '
          f'literature ground truth; {unclear} unclear and {missing} absent, '
          f'both defaulted to excitatory')
    for k, v in cls.value_counts().items():
        print(f'    {k:16s} {v:7d}  sign {NT_SIGN[k]:+d}')

    w = pd.read_feather(SRC / WTS)[['body_pre', 'body_post', 'weight']]
    keep = w.body_pre.isin(index_of.index) & w.body_post.isin(index_of.index)
    w = w[keep]
    print(f'\nconnections: {len(w)} rows among traced neurons '
          f'({int((~keep).sum())} dropped for touching an untraced body), '
          f'{int(w.weight.sum())} synapses')

    pre_sign = sign.reindex(w.body_pre.to_numpy()).to_numpy()
    out = pd.DataFrame({
        'Presynaptic_ID': w.body_pre.to_numpy(),
        'Postsynaptic_ID': w.body_post.to_numpy(),
        'Presynaptic_Index': index_of.loc[w.body_pre].to_numpy(),
        'Postsynaptic_Index': index_of.loc[w.body_post].to_numpy(),
        'Connectivity': w.weight.to_numpy(),
        'Excitatory': pre_sign,
    })
    out['Excitatory x Connectivity'] = out.Connectivity * out.Excitatory

    # A type map so experiments defined against FlyWire can be resolved here.
    # The stimulus is not guaranteed to be the same set of cells: FlyWire's
    # sugar experiment names 21 right-hemisphere neurons of type LB3 and this
    # volume has 87 of them, so the runner reports the count it resolved.
    types = (traced.set_index('bodyId')[['flywireType', 'type', 'superclass']]
             .reindex(ids))
    types.index.name = 'bodyId'
    types.to_csv(out_comp.with_name('malecns_types.csv'))
    print(f"wrote {out_comp.with_name('malecns_types.csv')}")

    pd.DataFrame({'Completed': np.ones(len(ids), dtype=int)},
                 index=pd.Index(ids, name='')).to_csv(out_comp)
    out.to_parquet(out_conn, index=False)
    print(f'\nwrote {out_comp}')
    print(f'wrote {out_conn}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-completeness', type=Path,
                    default=ROOT / 'data/malecns_completeness.csv')
    ap.add_argument('--out-connectivity', type=Path,
                    default=ROOT / 'data/malecns_connectivity.parquet')
    a = ap.parse_args()
    build(a.out_completeness, a.out_connectivity)


if __name__ == '__main__':
    main()
