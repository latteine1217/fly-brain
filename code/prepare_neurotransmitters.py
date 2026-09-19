"""
Build a per-neuron neurotransmitter table for the FlyWire v783 model neurons.

The connectivity parquet shipped with this repository has already collapsed
neurotransmitter identity into a single ``Excitatory`` column of +1/-1, applied
per neuron under Dale's law. That collapse is lossy in two ways:

  1. Dopaminergic, serotonergic, octopaminergic and tyraminergic neurons are
     neither GABAergic nor glutamatergic, so the majority-vote rule in Shiu et
     al. assigns them +1 and they become indistinguishable from cholinergic
     neurons.
  2. The Eckstein et al. (2024) classifier predicts six transmitters and cannot
     emit histamine at all. Histamine gates chloride channels (Ort/hclA on
     lamina monopolar cells, HisCl1 between photoreceptors) and is therefore
     inhibitory, but histaminergic neurons are scored +1 by the same rule.

This script recovers the discarded identity from the FlyWire annotations and
writes it alongside the model's own neuron index, so downstream code can either
correct the sign or give each transmitter its own synaptic kinetics.

Nothing here is inferred: every assignment is traced to either the published
classifier output (``top_nt``) or a literature source (``known_nt``), and the
provenance is recorded per neuron in ``nt_source``.

Usage:
    python code/prepare_neurotransmitters.py
    python code/prepare_neurotransmitters.py --annotations path/to/file.tsv
"""

import argparse
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ANNOTATIONS_URL = (
    'https://raw.githubusercontent.com/flyconnectome/flywire_annotations/'
    'main/supplemental_files/Supplemental_file1_neuron_annotations.tsv'
)

# Small-molecule transmitters that gate a fast ionotropic receptor, plus the
# monoamines. Only these determine nt_class; everything else in known_nt is a
# neuropeptide, a gaseous messenger, or a negative result.
CLASSICAL_NT = frozenset({
    'acetylcholine', 'glutamate', 'gaba', 'histamine', 'glycine',
    'dopamine', 'serotonin', 'octopamine', 'tyramine',
})

# Sign of the fast postsynaptic current, where one exists.
#   acetylcholine  nicotinic receptors, cation-selective          -> excitatory
#   glutamate      GluCl-alpha, chloride-selective in insect CNS  -> inhibitory
#   gaba           Rdl, chloride-selective                        -> inhibitory
#   histamine      Ort/hclA and HisCl1, chloride-selective        -> inhibitory
#   glycine        chloride-selective                             -> inhibitory
# The monoamines act through G-protein-coupled receptors and have no fixed fast
# sign; they are left as 0 so that callers must decide explicitly rather than
# inherit a default silently.
NT_SIGN = {
    'acetylcholine': 1,
    'glutamate': -1,
    'gaba': -1,
    'histamine': -1,
    'glycine': -1,
    'dopamine': 0,
    'serotonin': 0,
    'octopamine': 0,
    'tyramine': 0,
}

MODULATORY = frozenset({'dopamine', 'serotonin', 'octopamine', 'tyramine'})


def parse_known_nt(raw):
    """Resolve a free-text known_nt string to one classical transmitter.

    The field mixes co-transmitters ("gaba, nitric oxide"), duplicated entries
    ("acetylcholine; acetylcholine"), negative results ("gaba-negative") and
    neuropeptides in one comma/semicolon separated string. Returns the single
    classical transmitter if exactly one is named, otherwise None -- an
    ambiguous record is not worth guessing at when a classifier prediction is
    already available.
    """
    if not isinstance(raw, str):
        return None, ()

    tokens = {t.strip().lower() for t in re.split(r'[;,]', raw) if t.strip()}
    # "<nt>-negative" states that the neuron is NOT that transmitter.
    tokens = {t for t in tokens if not t.endswith('-negative') and t != 'negative'}

    classical = tokens & CLASSICAL_NT
    others = tuple(sorted(tokens - CLASSICAL_NT))
    if len(classical) == 1:
        return next(iter(classical)), others
    return None, others


def build_table(annotations_path, completeness_path):
    """Join FlyWire annotations onto the model's neuron index."""
    ann = pd.read_csv(annotations_path, sep='\t', low_memory=False)
    ann['root_id'] = ann['root_id'].astype('int64')

    comp = pd.read_csv(completeness_path, index_col=0)
    # Row order in the completeness file defines the model's tensor indices.
    model = pd.DataFrame({
        'flywire_id': comp.index.astype('int64'),
        'neuron_index': range(len(comp)),
    })

    cols = ['root_id', 'top_nt', 'top_nt_conf', 'known_nt', 'known_nt_source']
    merged = model.merge(
        ann[cols].drop_duplicates(subset='root_id'),
        left_on='flywire_id', right_on='root_id', how='left',
    ).drop(columns='root_id')

    parsed = merged['known_nt'].apply(parse_known_nt)
    merged['literature_nt'] = [p[0] for p in parsed]
    merged['cotransmitters'] = ['|'.join(p[1]) for p in parsed]

    # Literature beats classifier; classifier beats nothing.
    merged['nt_class'] = (
        merged['literature_nt']
        .fillna(merged['top_nt'].str.lower())
        .fillna('unknown')
    )
    merged['nt_source'] = 'unknown'
    merged.loc[merged['top_nt'].notna(), 'nt_source'] = 'predicted'
    merged.loc[merged['literature_nt'].notna(), 'nt_source'] = 'literature'

    merged['nt_sign'] = merged['nt_class'].map(NT_SIGN).fillna(0).astype(int)
    merged['is_modulatory'] = merged['nt_class'].isin(MODULATORY)

    return merged[[
        'flywire_id', 'neuron_index', 'nt_class', 'nt_source', 'nt_sign',
        'is_modulatory', 'top_nt', 'top_nt_conf', 'known_nt_source',
        'cotransmitters',
    ]]


def summarise(table, out_path):
    """Print the provenance and class breakdown the caller needs to judge this."""
    n = len(table)
    print(f'\nwrote {out_path}  ({n} neurons)')

    print('\nprovenance:')
    for src, cnt in table['nt_source'].value_counts().items():
        print(f'  {src:12s} {cnt:7d}  ({cnt / n * 100:5.2f}%)')

    print('\nneurotransmitter class:')
    for cls, cnt in table['nt_class'].value_counts().items():
        sign = NT_SIGN.get(cls)
        label = {1: 'excitatory', -1: 'inhibitory', 0: 'modulatory'}.get(sign, 'unknown')
        print(f'  {cls:16s} {cnt:7d}  ({cnt / n * 100:5.2f}%)  {label}')

    n_mod = int(table['is_modulatory'].sum())
    print(f'\nmodulatory neurons (no fast ionotropic sign): {n_mod}')


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--annotations', type=Path,
        default=root / 'data' / 'flywire_annotations_783.tsv',
        help='FlyWire annotation TSV; downloaded if absent',
    )
    parser.add_argument(
        '--completeness', type=Path,
        default=root / 'data' / '2025_Completeness_783.csv',
    )
    parser.add_argument(
        '--out', type=Path,
        default=root / 'data' / 'neurotransmitters_783.csv',
    )
    args = parser.parse_args()

    if not args.annotations.exists():
        print(f'downloading annotations to {args.annotations} ...')
        args.annotations.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ANNOTATIONS_URL, args.annotations)

    if not args.completeness.exists():
        sys.exit(f'completeness file not found: {args.completeness}')

    table = build_table(args.annotations, args.completeness)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    summarise(table, args.out)


if __name__ == '__main__':
    main()
