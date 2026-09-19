"""
Per-neurotransmitter synaptic channels for the Drosophila LIF brain model.

The published model gives every synapse one time constant (5 ms) and one of two
signs. This module keeps the same LIF neuron but splits the synaptic input into
one conductance per distinct time constant, so that transmitters with different
receptor kinetics decay at different rates.

What is and is not grounded here matters, so it is stated plainly:

  * The per-neuron transmitter assignment is data. It comes from
    ``data/neurotransmitters_783.csv`` (see prepare_neurotransmitters.py),
    which traces every neuron to a literature source or to the published
    Eckstein et al. (2024) classifier.

  * The sign of each fast ionotropic current is textbook and cited below.

  * The time constants are mostly NOT constrained. Only acetylcholine has a
    measured decay constant usable here. Every other entry keeps the paper's
    5 ms and is marked UNCONSTRAINED. They are deliberately left at the
    published value rather than filled with plausible-looking numbers, so that
    enabling this model changes exactly one parameter and any difference in
    behaviour is attributable.

Channels are grouped by time constant, not by transmitter: transmitters sharing
a tau share a conductance, because their contributions are summed anyway and
the sign is already folded into the weights. With the default table that means
two channels. If every tau is equal the model collapses to one channel and
reproduces the published model exactly.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================================
# Synaptic kinetics per neurotransmitter
# ============================================================================

# tau    decay time constant of the postsynaptic conductance, in ms
# sign   +1 excitatory, -1 inhibitory, 0 no fast ionotropic current
# source provenance of the tau value; UNCONSTRAINED means no usable
#        measurement was found for a central Drosophila synapse
NT_KINETICS = {
    'acetylcholine': {
        'tau': 2.0, 'sign': 1,
        'source': "Lee & O'Dowd 1999, J Neurosci 19(13):5311-5321 -- mEPSC "
                  "decay tau 2 ms, rise 0.6 ms, cultured embryonic neurons "
                  "3-9 DIV. Not adult central brain.",
    },
    'gaba': {
        'tau': 5.0, 'sign': -1,
        'source': 'UNCONSTRAINED. Wilson & Laurent 2005 (J Neurosci 25:9069) '
                  'separate a fast picrotoxin-sensitive GABA-A component from '
                  'a slow GABA-B component of tens to thousands of ms present '
                  'only in projection neurons. Which applies depends on '
                  'postsynaptic receptor expression, which the connectome does '
                  'not carry. Left at the published 5 ms.',
    },
    'glutamate': {
        'tau': 5.0, 'sign': -1,
        'source': 'UNCONSTRAINED. GluCl-alpha is the inhibitory glutamate '
                  'receptor of the insect CNS, but the decay constants in '
                  'circulation (2.5 s, 6.5 s) are desensitisation constants '
                  'from Xenopus oocyte expression under sustained agonist, not '
                  'synaptic decay. Left at the published 5 ms.',
    },
    'histamine': {
        'tau': 5.0, 'sign': -1,
        'source': 'UNCONSTRAINED tau. Sign is established: Ort/hclA and HisCl1 '
                  'are histamine-gated chloride channels (Gengs et al. 2002, '
                  'JBC 277:42113), so photoreceptor output is inhibitory. '
                  'Left at the published 5 ms.',
    },
    'glycine': {
        'tau': 5.0, 'sign': -1,
        'source': 'UNCONSTRAINED. 16 neurons. Left at the published 5 ms.',
    },
    # Monoamines act through G-protein-coupled receptors on second-to-minute
    # timescales that a 0.1 ms fixed-step alpha synapse cannot represent, and
    # the receptor expression map is not in the connectome. Sign 0 means the
    # shipped Excitatory value is kept rather than a sign being invented.
    'dopamine': {'tau': 5.0, 'sign': 0, 'source': 'UNCONSTRAINED. Metabotropic; see module docstring.'},
    'serotonin': {'tau': 5.0, 'sign': 0, 'source': 'UNCONSTRAINED. Metabotropic; see module docstring.'},
    'octopamine': {'tau': 5.0, 'sign': 0, 'source': 'UNCONSTRAINED. Metabotropic; see module docstring.'},
    'tyramine': {'tau': 5.0, 'sign': 0, 'source': 'UNCONSTRAINED. Metabotropic; see module docstring.'},
    'unknown': {'tau': 5.0, 'sign': 0, 'source': 'No transmitter assignment available.'},
}


# ============================================================================
# Weight construction
# ============================================================================

def load_nt_table(nt_path, num_neurons):
    """Load the per-neuron transmitter table, ordered by model neuron index."""
    nt = pd.read_csv(nt_path)
    if len(nt) != num_neurons:
        raise ValueError(
            f'neurotransmitter table has {len(nt)} rows but the model has '
            f'{num_neurons} neurons; regenerate it with '
            f'code/prepare_neurotransmitters.py'
        )
    nt = nt.sort_values('neuron_index')
    unknown = set(nt['nt_class']) - set(NT_KINETICS)
    if unknown:
        raise ValueError(f'transmitter classes with no kinetics entry: {sorted(unknown)}')
    return nt


def resolve_signs(nt, shipped_sign):
    """Sign per presynaptic neuron, overriding only where a fast sign exists.

    Neurons whose transmitter has no fast ionotropic current (the monoamines,
    and neurons with no assignment) keep the sign shipped with the connectivity
    data. Overriding those would mean inventing a polarity for a metabotropic
    synapse.
    """
    shipped = np.asarray(shipped_sign, dtype=np.int64)
    sign = shipped.copy()
    fast = nt['nt_class'].map(lambda c: NT_KINETICS[c]['sign']).to_numpy()
    override = fast != 0
    sign[override] = fast[override]
    # Report neurons whose polarity actually flipped, not every neuron the rule
    # touched -- most of those already agreed with the shipped value.
    return sign, int((sign != shipped).sum())


def build_channels(conn, nt, num_neurons, shipped_sign, force_tau=None,
                   charge_ref_tau=None):
    """Group transmitters by time constant and build one sparse matrix each.

    Returns (taus, matrices, report) where matrices[k] is a sparse COO matrix
    of shape (num_neurons, num_neurons) holding only the synapses whose
    presynaptic neuron belongs to channel k.

    force_tau collapses every transmitter onto one time constant, which yields
    a single matrix carrying the corrected signs and nothing else. That is the
    sign-only correction: it changes the data without changing the model.
    """
    sign, n_override = resolve_signs(nt, shipped_sign)

    # Channel identity is the time constant, so transmitters that share a tau
    # share a conductance. Sorted for a deterministic channel order.
    tau_of_class = {
        c: (force_tau if force_tau is not None else NT_KINETICS[c]['tau'])
        for c in nt['nt_class'].unique()
    }
    taus = sorted(set(tau_of_class.values()))
    channel_of_class = {c: taus.index(t) for c, t in tau_of_class.items()}

    channel_of_neuron = nt['nt_class'].map(channel_of_class).to_numpy()

    pre = conn['Presynaptic_Index'].to_numpy()
    post = conn['Postsynaptic_Index'].to_numpy()
    # Rebuild the weight from the raw synapse count and the resolved sign,
    # rather than reusing the pre-signed column.
    val = conn['Connectivity'].to_numpy() * sign[pre]

    syn_channel = channel_of_neuron[pre]
    matrices, report = [], []
    for k, tau in enumerate(taus):
        m = syn_channel == k
        # In this alpha synapse the conductance decays as exp(-t/tau) from the
        # input amplitude, so one presynaptic spike transfers charge
        # proportional to tau. Left alone, giving acetylcholine a shorter tau
        # weakens excitation relative to inhibition and moves the network off
        # its operating point for reasons that have nothing to do with
        # kinetics. Scaling by charge_ref_tau/tau holds the charge per spike
        # fixed so that only the time course differs.
        gain = 1.0 if charge_ref_tau is None else charge_ref_tau / tau
        idx = torch.from_numpy(np.stack([post[m], pre[m]]))
        matrices.append(
            torch.sparse_coo_tensor(
                idx, torch.from_numpy(val[m] * gain).to(torch.float32),
                (num_neurons, num_neurons),
            ).coalesce()
        )
        classes = sorted(c for c, ch in channel_of_class.items() if ch == k)
        report.append({
            'tau': tau, 'gain': gain, 'classes': classes,
            'neurons': int((channel_of_neuron == k).sum()),
            'synapses': int(m.sum()),
        })

    return taus, matrices, {'channels': report, 'signs_overridden': n_override}


# ============================================================================
# Model
# ============================================================================

class MultiNTTorchModel(nn.Module):
    """LIF neuron driven by one alpha conductance per synaptic time constant.

    The delay line holds spikes rather than weighted input. Because the weight
    matrix is linear and the 1.8 ms delay is uniform across synapses, delaying
    spikes and then weighting is identical to weighting and then delaying, and
    it keeps one delay buffer instead of one per channel.
    """

    def __init__(self, batch, size, dt, params, taus, matrices,
                 neuron, exc_indices=None, device='cpu'):
        super().__init__()
        self.batch, self.size, self.dt = batch, size, dt
        self.neuron = neuron
        self.weights = matrices
        self.n_channels = len(matrices)
        self.scale = params['wScale']
        self.steps_delay = int(params['tDelay'] / dt)

        # (K, 1) so it broadcasts over (B, K, N)
        self.decay = torch.tensor(
            [1.0 - dt / t for t in taus], dtype=torch.float32, device=device
        ).view(1, -1, 1)

        self.poisson_scale = params['scalePoisson']
        self.prob_scale = dt / 1000.0

        base_refrac = int(round(params['tRefrac'] / dt))
        self.refrac_steps = torch.full((size,), base_refrac, dtype=torch.long, device=device)
        if exc_indices is not None:
            self.refrac_steps[exc_indices] = 0
        self.device = device

    def state_init(self):
        d = self.device
        conductance = torch.zeros(self.batch, self.n_channels, self.size, device=d)
        delay_buffer = torch.zeros(self.batch, self.steps_delay + 1, self.size, device=d)
        spikes = torch.zeros(self.batch, self.size, device=d)
        v = torch.zeros(self.batch, self.size, device=d) + self.neuron.v_0
        refrac = self.refrac_steps.unsqueeze(0).repeat(self.batch, 1).float()
        return conductance, delay_buffer, spikes, v, refrac

    def forward(self, rates, conductance, delay_buffer, spikes, v, refrac,
                generator=None):
        poisson_spikes = torch.bernoulli(
            rates * self.prob_scale, generator=generator
        ) * self.poisson_scale
        voltage_stim = self.scale * poisson_spikes

        refrac = torch.where(spikes > 0, torch.zeros_like(refrac), refrac + 1)
        gate = (refrac >= self.refrac_steps.unsqueeze(0)).float()

        delayed = delay_buffer[:, 0, :]
        delay_buffer = torch.roll(delay_buffer, shifts=-1, dims=1)
        delay_buffer[:, -1, :] = spikes

        # One sparse matmul per channel; total non-zeros across channels equals
        # the full connectome, so this is a partition of the same work.
        drive = torch.stack(
            [torch.sparse.mm(w, delayed.t().contiguous()).t() for w in self.weights],
            dim=1,
        )
        conductance_new = conductance * self.decay + self.scale * drive * gate.unsqueeze(1)

        spikes, v_new = self.neuron(conductance.sum(dim=1), voltage_stim, v)

        conductance_new = conductance_new - (conductance_new * spikes.unsqueeze(1)).detach()
        return conductance_new, delay_buffer, spikes, v_new, refrac
