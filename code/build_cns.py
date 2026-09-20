"""
Join the FlyWire brain to the MANC ventral nerve cord.

A descending neuron is one cell with its soma and dendrites in the brain and
its axon in the nerve cord. FlyWire sees the first half and stops at the neck;
MANC sees the second half and starts there. Joining the two datasets is
therefore not a matter of wiring two networks together but of sewing severed
neurons back into one, so that a descending neuron driven in the brain delivers
its spikes to the motor circuits that actually move the animal.

Two things make this possible and one thing makes it approximate.

The published cross-dataset work (Stürner et al., the neck connective paper)
harmonised descending neuron type names across FAFB/FlyWire, FANC and MANC, and
its supplemental tables carry both a FlyWire root_id and a MANC bodyId under
that shared vocabulary. Matching on (type, side) pairs 394 groups covering 532
FlyWire and 536 MANC descending neurons. MANC also ships a neurotransmitter
prediction for 99.4% of its traced neurons, which is what the model needs to
give each one a sign.

What stays approximate: FlyWire is a female brain and MANC a male nerve cord,
so this is a chimera. Only about 41% of descending neurons could be matched;
the rest keep their nerve cord half, driven by the cord's own circuits, and
their brain half, driven by the brain, with nothing joining them. Both numbers
are reported when the file is built rather than buried.

Outputs are written in exactly the format the existing model reads, so nothing
downstream needs to know the nerve cord is there:
    data/cns_completeness.csv
    data/cns_connectivity.parquet

Usage:
    python code/build_cns.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Same convention as the brain model: acetylcholine excites, the chloride-gated
# transmitters inhibit. A neuron with no usable prediction takes the excitatory
# default that Shiu et al.'s majority rule would also have given it.
NT_SIGN = {
    'acetylcholine': 1,
    'gaba': -1,
    'glutamate': -1,
    'unknown': 1,
}


def load_brain():
    comp = pd.read_csv(ROOT / 'data/2025_Completeness_783.csv', index_col=0)
    conn = pd.read_parquet(ROOT / 'data/2025_Connectivity_783.parquet')
    return comp, conn


def load_cord():
    neu = pd.read_csv(ROOT / 'data/manc/traced-neurons.csv')
    conn = pd.read_csv(ROOT / 'data/manc/traced-connections.csv')
    props = pd.read_feather(ROOT / 'data/manc/manc-v1.0-neuron-properties.feather')
    neu = neu.merge(props[['bodyId', 'predictedNt']], on='bodyId', how='left')
    return neu, conn


def match_descending(brain_ids):
    """Pair FlyWire descending neurons with their nerve cord halves.

    Returns a Series mapping MANC bodyId -> FlyWire root_id, and a report.

    Groups are keyed on (type, side) because descending neurons come in
    bilateral pairs and a left cell must not inherit a right axon. Where a
    group holds several cells on each side they are members of one type, so
    MANC cells are dealt round-robin onto the FlyWire cells of that group:
    total synaptic weight is preserved and no cell is duplicated.
    """
    fa = pd.read_csv(ROOT / 'data/vnc/Supplemental_file5_FAFB_DNs.tsv',
                     sep='\t', low_memory=False)
    ma = pd.read_csv(ROOT / 'data/vnc/Supplemental_file7_MANC_DNs.tsv',
                     sep='\t', low_memory=False)
    fa['root_id'] = pd.to_numeric(fa['root_id'], errors='coerce')
    fa = fa[fa.root_id.isin(brain_ids) & fa.type.notna() & fa.side.notna()]
    ma = ma[ma.type.notna() & ma.side.notna()]

    pairs, groups = {}, 0
    fa_groups = fa.groupby(['type', 'side'])
    for key, m_grp in ma.groupby(['type', 'side']):
        if key not in fa_groups.groups:
            continue
        f_ids = fa_groups.get_group(key).root_id.astype('int64').tolist()
        groups += 1
        for i, body in enumerate(m_grp.bodyid.astype('int64')):
            pairs[body] = f_ids[i % len(f_ids)]

    report = {
        'groups': groups,
        'manc_dns_matched': len(pairs),
        'fafb_dns_matched': len(set(pairs.values())),
        'manc_dns_total': int(ma.bodyid.nunique()),
        'fafb_dns_total': int(fa.root_id.nunique()),
    }
    return pd.Series(pairs, dtype='int64'), report


def build(out_comp, out_conn):
    comp, brain_conn = load_brain()
    brain_ids = comp.index.astype('int64').to_numpy()
    n_brain = len(brain_ids)
    print(f'brain: {n_brain} neurons, {len(brain_conn)} connections')

    cord_neu, cord_conn = load_cord()
    print(f'cord : {len(cord_neu)} neurons, {len(cord_conn)} connections')

    pairs, rep = match_descending(set(brain_ids.tolist()))
    print(f"\nmatched {rep['groups']} (type, side) groups: "
          f"{rep['fafb_dns_matched']} brain descending neurons "
          f"({rep['fafb_dns_matched'] / rep['fafb_dns_total'] * 100:.1f}%) "
          f"joined to {rep['manc_dns_matched']} cord counterparts "
          f"({rep['manc_dns_matched'] / rep['manc_dns_total'] * 100:.1f}%)")

    # Combined neuron list: the brain keeps its indices, so anything built
    # against the brain-only files still refers to the same neurons.
    merged = set(pairs.index)
    cord_own = cord_neu[~cord_neu.bodyId.isin(merged)].copy()
    all_ids = np.concatenate([brain_ids, cord_own.bodyId.to_numpy(dtype='int64')])
    if len(set(all_ids.tolist())) != len(all_ids):
        sys.exit('id collision between FlyWire root ids and MANC body ids')
    index_of = pd.Series(np.arange(len(all_ids)), index=all_ids)
    print(f'combined: {len(all_ids)} neurons '
          f'({len(cord_own)} added, {len(merged)} sewn onto brain cells)')

    # Signs. Brain neurons keep theirs; cord neurons take the prediction.
    brain_sign = brain_conn.groupby('Presynaptic_ID')['Excitatory'].first()
    cord_sign = (cord_neu.set_index('bodyId')['predictedNt']
                 .str.lower().map(NT_SIGN).fillna(1).astype(int))
    unresolved = int(cord_neu.predictedNt.isna().sum()
                     + (cord_neu.predictedNt.str.lower() == 'unknown').sum())
    print(f'cord signs: {int((cord_sign > 0).sum())} excitatory, '
          f'{int((cord_sign < 0).sum())} inhibitory '
          f'({unresolved} defaulted for want of a prediction)')

    # A sewn descending neuron keeps the brain's sign: it is one cell, and the
    # brain dataset is this model's reference for transmitter identity.
    disagree = sum(1 for b, f in pairs.items()
                   if b in cord_sign.index and f in brain_sign.index
                   and cord_sign[b] != brain_sign[f])
    print(f'  {disagree} of {len(pairs)} sewn pairs disagree on sign; '
          f'the brain assignment is kept')

    # Cord connections, re-indexed onto the combined neuron list.
    #
    # Deliberately not Series.map: a partial map yields NaN, which forces the
    # column to float64, and a FlyWire root id is ~7.2e17 against float64's
    # exact-integer ceiling of 2^53. The ids come back silently rounded and
    # the sewing quietly fails to happen, while everything downstream still
    # reports plausible totals.
    cc = cord_conn.copy()
    keys = pairs.index.to_numpy()
    order = np.argsort(keys)
    keys, vals = keys[order], pairs.to_numpy()[order]

    def sew(body_ids):
        """Replace a matched cord body id with its brain counterpart."""
        pos = np.clip(np.searchsorted(keys, body_ids), 0, len(keys) - 1)
        return np.where(keys[pos] == body_ids, vals[pos], body_ids).astype('int64')

    cc['pre_id'] = sew(cc.bodyId_pre.to_numpy(dtype='int64'))
    cc['post_id'] = sew(cc.bodyId_post.to_numpy(dtype='int64'))

    sewn_rows = int(cc.bodyId_pre.isin(pairs.index).sum())
    got = int((cc.pre_id != cc.bodyId_pre).sum())
    assert got == sewn_rows, (
        f'{got} rows were re-pointed at a brain neuron but {sewn_rows} cord '
        f'edges start at a matched descending neuron'
    )

    cc = cc[cc.pre_id.isin(index_of.index) & cc.post_id.isin(index_of.index)]
    sign = cc.pre_id.map(brain_sign).fillna(cc.bodyId_pre.map(cord_sign)).fillna(1)

    cord_rows = pd.DataFrame({
        'Presynaptic_ID': cc.pre_id.to_numpy(),
        'Postsynaptic_ID': cc.post_id.to_numpy(),
        'Presynaptic_Index': index_of.loc[cc.pre_id].to_numpy(),
        'Postsynaptic_Index': index_of.loc[cc.post_id].to_numpy(),
        'Connectivity': cc.weight.to_numpy(),
        'Excitatory': sign.to_numpy().astype('int64'),
    })
    cord_rows['Excitatory x Connectivity'] = (
        cord_rows.Connectivity * cord_rows.Excitatory)

    combined = pd.concat([brain_conn, cord_rows], ignore_index=True)
    # Sewing can produce two rows for one ordered pair; the model reads a
    # weight per pair, so they are summed rather than left to collide.
    combined = (combined.groupby(
        ['Presynaptic_ID', 'Postsynaptic_ID', 'Presynaptic_Index',
         'Postsynaptic_Index', 'Excitatory'], as_index=False)
        .agg({'Connectivity': 'sum'}))
    combined['Excitatory x Connectivity'] = (
        combined.Connectivity * combined.Excitatory)

    print(f'\ncombined connectivity: {len(combined)} rows '
          f'({len(combined) - len(brain_conn):+d} vs brain alone), '
          f'{int(combined.Connectivity.sum())} synapses')

    pd.DataFrame({'Completed': np.ones(len(all_ids), dtype=int)},
                 index=pd.Index(all_ids, name='')).to_csv(out_comp)
    combined.to_parquet(out_conn, index=False)
    print(f'wrote {out_comp}')
    print(f'wrote {out_conn}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-completeness', type=Path,
                    default=ROOT / 'data/cns_completeness.csv')
    ap.add_argument('--out-connectivity', type=Path,
                    default=ROOT / 'data/cns_connectivity.parquet')
    args = ap.parse_args()
    build(args.out_completeness, args.out_connectivity)


if __name__ == '__main__':
    main()
