"""
PyTorch benchmark runner for the Drosophila brain model.

Implements the LIF neuron model with alpha-function synapses using PyTorch,
with support for CPU, CUDA and Apple Metal (MPS) computation. Batches n_run
trials in parallel for efficient GPU utilization.

Model architecture (from Shiu et al.):
    PoissonSpikeGenerator → recurrent weights (sparse matmul) → AlphaLIF
    where AlphaLIF = AlphaSynapse + LIFNeuron + refractory period

Called by benchmark.py orchestrator.
"""

import os
import pandas as pd
import pyarrow  # noqa: F401  — must be imported before torch to avoid libarrow conflict
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from time import perf_counter as time
import traceback

from model_multi_nt import MultiNTTorchModel, build_channels, load_nt_table
from propagation import SparsePropagation, build as build_propagation
from benchmark import (
    T_RUN_VALUES_SEC, N_RUN_VALUES,
    path_comp, path_con, path_wt,
    get_experiment, get_spike_output_path, print_summary_table, save_result_csv,
    spike_io_enabled,
)

# ============================================================================
# PyTorch Model Parameters (matching Brian2 default_params)
# ============================================================================

MODEL_PARAMS = {
    'tauSyn': 5.0,        # ms
    'tDelay': 1.8,        # ms
    'v0': -52.0,          # mV
    'vReset': -52.0,      # mV
    'vRest': -52.0,       # mV
    'vThreshold': -45.0,  # mV
    'tauMem': 20.0,       # ms
    'tRefrac': 2.2,       # ms
    'scalePoisson': 250,
    'wScale': 0.275,
}

DT = 0.1  # Simulation timestep in ms (matches Brian2 defaultclock.dt)

# ============================================================================
# Device Selection
# ============================================================================

# Set to cpu/cuda/mps to pin the backend. The MPS-vs-CPU parity check relies on
# this: the two runs must differ in the device and in nothing else.
DEVICE_ENV_VAR = 'FLYBRAIN_TORCH_DEVICE'

VALID_DEVICES = ('cpu', 'cuda', 'mps')


def resolve_device():
    """Select the compute device, honouring an explicit override.

    Fails loudly when an override names an unavailable backend: silently
    demoting to CPU would turn a device benchmark into a mislabelled CPU run.
    """
    override = os.environ.get(DEVICE_ENV_VAR, '').strip().lower()
    if override:
        if override not in VALID_DEVICES:
            raise ValueError(
                f'{DEVICE_ENV_VAR} must be one of {VALID_DEVICES}, got {override!r}'
            )
        if override == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(f'{DEVICE_ENV_VAR}=cuda but CUDA is unavailable')
        if override == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError(f'{DEVICE_ENV_VAR}=mps but MPS is unavailable')
        return override

    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


NT_MODE_ENV_VAR = 'FLYBRAIN_NT_MODE'

# paper    : the published model -- one time constant, the shipped +1/-1 signs
# signs    : the published model with transmitter signs corrected from the
#            FlyWire annotations (histamine and glycine are inhibitory; a
#            classifier majority vote scored them excitatory)
# channels : one synaptic conductance per distinct time constant
# channels-charge
#          : as channels, with each channel rescaled to hold the charge per
#            presynaptic spike fixed. Raw 'channels' cannot be returned to the
#            paper's operating point by w_syn alone -- MN9's rate is not
#            monotonic in w_syn there, because shortening only the cholinergic
#            time constant shifts the excitation/inhibition balance.
VALID_NT_MODES = ('paper', 'signs', 'channels', 'channels-charge')


PROPAGATION_ENV_VAR = 'FLYBRAIN_PROPAGATION'

# sparse : the published formulation, one sparse matrix product per timestep
# event  : gather only the outgoing synapses of neurons that fired
VALID_PROPAGATIONS = ('sparse', 'event')


SILENCE_ENV_VAR = 'FLYBRAIN_SILENCE'


def resolve_extra_silenced():
    """Additional FlyWire ids to silence, on top of the experiment's own list.

    Lesioning is how a control is run against this model, and an experiment
    definition lives in source, so without this the tool can only be used by
    editing benchmark.py.
    """
    raw = os.environ.get(SILENCE_ENV_VAR, '').strip()
    if not raw:
        return []
    ids = []
    for token in raw.replace(',', ' ').split():
        try:
            ids.append(int(token))
        except ValueError:
            raise ValueError(
                f'{SILENCE_ENV_VAR} takes FlyWire root ids separated by commas '
                f'or spaces; could not read {token!r}'
            ) from None
    return ids


SETTLE_CHECK_ENV_VAR = 'FLYBRAIN_SETTLE_CHECK'

# The quiet period run after the stimulus, and how it is reported.
SETTLE_WINDOWS = 4
SETTLE_WINDOW_MS = 25.0


def resolve_settle_check():
    """Whether to test that the network returns to rest after the stimulus.

    On by default. The check runs outside the timed section, so benchmark
    numbers are unaffected and only wall-clock grows.
    """
    return os.environ.get(SETTLE_CHECK_ENV_VAR, '').strip().lower() not in (
        '0', 'false', 'no', 'off'
    )


def resolve_propagation():
    """Select how spikes are pushed through the connectome.

    Both give identical output, so this is purely a cost choice; the default
    keeps the published formulation.
    """
    mode = os.environ.get(PROPAGATION_ENV_VAR, '').strip().lower() or 'sparse'
    if mode not in VALID_PROPAGATIONS:
        raise ValueError(
            f'{PROPAGATION_ENV_VAR} must be one of {VALID_PROPAGATIONS}, '
            f'got {mode!r}'
        )
    return mode


COMPILE_ENV_VAR = 'FLYBRAIN_COMPILE'


DATASET_ENV_VAR = 'FLYBRAIN_DATASET'

# brain : FlyWire v783, 138,639 neurons, the published model
# cns   : that brain sewn to the MANC ventral nerve cord, 161,291 neurons,
#         so descending commands reach motor neurons (see code/build_cns.py)
VALID_DATASETS = ('brain', 'cns')


def resolve_dataset():
    """Which connectome to simulate."""
    name = os.environ.get(DATASET_ENV_VAR, '').strip().lower() or 'brain'
    if name not in VALID_DATASETS:
        raise ValueError(
            f'{DATASET_ENV_VAR} must be one of {VALID_DATASETS}, got {name!r}'
        )
    return name


def dataset_paths(name):
    """Connectivity, completeness and weight-cache locations for a dataset.

    Each dataset caches its prepared weight matrices in its own directory:
    they share a filename, and one silently standing in for the other would be
    hard to notice and produce a wrong answer rather than an error.
    """
    if name == 'brain':
        return Path(path_con), Path(path_comp), Path(path_wt)
    data = Path(path_wt)
    cache = data / 'cns_weights'
    cache.mkdir(parents=True, exist_ok=True)
    conn, comp = data / 'cns_connectivity.parquet', data / 'cns_completeness.csv'
    for p in (conn, comp):
        if not p.exists():
            raise FileNotFoundError(
                f'{p} not found; build it with python code/build_cns.py'
            )
    return conn, comp, cache


def resolve_compile():
    """Whether to put the model through torch.compile.

    Output is bit-identical either way, so this only trades a one-off
    compilation for a cheaper step.
    """
    return os.environ.get(COMPILE_ENV_VAR, '').strip().lower() in (
        '1', 'true', 'yes', 'on'
    )


def resolve_nt_mode():
    """Select the synapse model. Defaults to reproducing the published one."""
    mode = os.environ.get(NT_MODE_ENV_VAR, '').strip().lower() or 'paper'
    if mode not in VALID_NT_MODES:
        raise ValueError(
            f'{NT_MODE_ENV_VAR} must be one of {VALID_NT_MODES}, got {mode!r}'
        )
    return mode


def synchronize(device_name):
    """Block until queued device work completes.

    CUDA and MPS both dispatch asynchronously, so without this the simulation
    timer stops while work is still in flight and reports a fictional duration.
    """
    if device_name == 'cuda':
        torch.cuda.synchronize()
    elif device_name == 'mps':
        torch.mps.synchronize()


# Device memory the spike raster may occupy between readbacks: one byte per
# neuron, per trial, per buffered step.
SPIKE_WINDOW_BYTES = 64 * 1024 ** 2


def spike_window_steps(batch, num_neurons):
    """How many steps of spike raster to buffer on the device at a time.

    Reading spikes back every step forces a host-device synchronisation every
    step; buffering a window and reading it back once removes all but one per
    window. Measured over ten interleaved pairs on MPS the saving is 0.282 ms
    per step (sd 0.085 ms, all ten pairs in the same direction), which is 2.7%
    of a 10.3 ms step there. It is a fixed synchronisation cost, so it is a
    larger fraction of a faster step.

    Timings on this device drift by more than this margin over minutes, so the
    comparison has to be paired and interleaved; two runs taken minutes apart
    cannot resolve it.
    """
    per_step = max(batch * num_neurons, 1)
    return max(1, min(1000, SPIKE_WINDOW_BYTES // per_step))


def memory_used_gb(device_name):
    """Device memory in use, or None where the backend cannot report it."""
    if device_name == 'cuda':
        free, total = torch.cuda.mem_get_info(device_name)
        return (total - free) / 1024 ** 3
    if device_name == 'mps':
        return torch.mps.current_allocated_memory() / 1024 ** 3
    return None


# ============================================================================
# Model Classes
# ============================================================================

class PoissonSpikeGenerator(nn.Module):
    """Generates one timestep of Poisson-distributed spikes from firing rates."""

    def __init__(self, dt, scale, device='cpu'):
        super().__init__()
        self.prob_scale = dt / 1000.0
        self.scale = scale
        self.device = device

    def forward(self, rates, generator=None):
        return torch.bernoulli(rates * self.prob_scale, generator=generator) * self.scale


class AlphaSynapse(nn.Module):
    """Alpha-function synapse dynamics with configurable delay."""

    def __init__(self, batch, size, dt, params, device='cpu'):
        super().__init__()
        self.time_factor = dt / params['tauSyn']
        self.steps_delay = int(params['tDelay'] / dt)
        self.size = size
        self.device = device
        self.batch = batch

    def state_init(self):
        conductance = torch.zeros(self.batch, self.size, device=self.device)
        delay_buffer = torch.zeros(
            self.batch, self.steps_delay + 1, self.size, device=self.device
        )
        return conductance, delay_buffer

    def forward(self, input_, conductance, delay_buffer, refrac):
        conductance_new = (
            conductance * (1 - self.time_factor) + delay_buffer[:, 0, :] * refrac
        )
        delay_buffer = torch.roll(delay_buffer, shifts=-1, dims=1)
        delay_buffer[:, -1, :] = input_
        return conductance_new, delay_buffer


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with surrogate gradient (ATan)."""

    def __init__(self, batch, size, dt, params, device='cpu'):
        super().__init__()
        self.size = size
        self.dt = dt
        self.tau_mem = params['tauMem']
        self.v_reset = params['vReset']
        self.v_rest = params['vRest']
        self.v_threshold = params['vThreshold']
        self.v_0 = params['v0']
        self.time_factor = dt / self.tau_mem
        self.spike_gradient = self.ATan.apply
        self.device = device
        self.batch = batch

    def state_init(self):
        v = torch.zeros(self.batch, self.size, device=self.device) + self.v_0
        spikes = torch.zeros(self.batch, self.size, device=self.device)
        return spikes, v

    def forward(self, conductance, voltage_stim, v):
        # Brian/GeNN:
        # V += Vstim
        # V += MemFactor * (G - (V - Vrest))

        v = v + voltage_stim
        v = v + self.time_factor * (conductance - (v - self.v_rest))

        spike = self.spike_gradient(v - self.v_threshold)

        reset = ((v - self.v_reset) * spike).detach()
        v = v - reset

        return spike, v

    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v):
            spike = (v > 0).float()
            ctx.save_for_backward(v)
            return spike

        @staticmethod
        def backward(ctx, grad_output):
            (v,) = ctx.saved_tensors
            grad = 1 / (1 + (np.pi * v).pow_(2)) * grad_output
            return grad


class AlphaLIF(nn.Module):
    """LIF neuron with alpha-function synapse dynamics and refractory period."""

    def __init__(
        self,
        batch,
        size,
        dt,
        params,
        exc_indices=None,
        device='cpu'
    ):
        super().__init__()
        self.size = size
        self.synapse = AlphaSynapse(batch, size, dt, params, device=device)
        self.neuron = LIFNeuron(batch, size, dt, params, device=device)
        base_refrac = int(round(params['tRefrac'] / dt))

        self.refrac_steps = torch.full(
            (size,),
            base_refrac,
            dtype=torch.long,
            device=device,
        )

        if exc_indices is not None:
            self.refrac_steps[exc_indices] = 0

    def state_init(self):
        conductance, delay_buffer = self.synapse.state_init()
        spikes, v = self.neuron.state_init()
        refrac = self.refrac_steps.unsqueeze(0).repeat(self.neuron.batch, 1).float()
        return conductance, delay_buffer, spikes, v, refrac

    def forward(self,
            recurrent_input,
            voltage_stim,
            conductance,
            delay_buffer,
            spikes,
            v,
            refrac):
        refrac = torch.where(
            spikes > 0,
            torch.zeros_like(refrac),
            refrac + 1,
        )
        conductance_new, delay_buffer = self.synapse(
            recurrent_input,
            conductance,
            delay_buffer,
            (
                refrac >= self.refrac_steps.unsqueeze(0)
            ).float()
        )

        spikes, v_new = self.neuron(
            conductance,
            voltage_stim,
            v
        )
        conductance_reset = (conductance_new * spikes).detach()
        conductance_new = conductance_new - conductance_reset
        return conductance_new, delay_buffer, spikes, v_new, refrac


class TorchModel(nn.Module):
    """
    Top-level model: Poisson input + recurrent connectome weights + AlphaLIF.

    The weights tensor should be a sparse matrix (CSR or COO) derived from
    the Drosophila connectome.
    """

    def __init__(
            self,
            batch,
            size,
            dt,
            params,
            weights,
            exc_indices=None,
            device='cpu',
            propagate=None
        ):
        super().__init__()
        self.neurons = AlphaLIF(
            batch,
            size,
            dt,
            params,
            exc_indices=exc_indices,
            device=device
        )
        self.weights = weights
        self.propagate = propagate if propagate is not None else SparsePropagation(weights)
        self.poisson = PoissonSpikeGenerator(dt, params['scalePoisson'], device=device)
        self.scale = params['wScale']

    def state_init(self):
        return self.neurons.state_init()

    def forward(self, rates, conductance, delay_buffer, spikes, v, refrac, generator=None):
        poisson_spikes = self.poisson(
            rates,
            generator=generator
        )

        voltage_stim = self.scale * poisson_spikes

        # See code/propagation.py: the default sparse product is written as
        # sparse.mm rather than matmul(dense, sparse.T) because the latter
        # leaves the optimised sparse kernel, and the event strategy skips the
        # multiply entirely for neurons that did not fire.
        weighted_spikes = self.propagate(spikes)

        recurrent_input = self.scale * weighted_spikes

        conductance, delay_buffer, spikes, v, refrac = self.neurons(
            recurrent_input,
            voltage_stim,
            conductance,
            delay_buffer,
            spikes,
            v,
            refrac,
        )
        return conductance, delay_buffer, spikes, v, refrac

# ============================================================================
# Data Utilities
# ============================================================================

def get_hash_tables(comp_path):
    """Build flywire ID <-> tensor index mappings from completeness CSV."""
    df_comp = pd.read_csv(comp_path, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    i2flyid = {j: i for i, j in flyid2i.items()}
    return flyid2i, i2flyid


def get_weights(conn_path, comp_path, wt_dir, csr=True):
    """Load or build sparse weight matrix from connectivity data.

    Caches weight_coo.pkl / weight_csr.pkl in wt_dir for reuse.
    """
    wt_dir = Path(wt_dir)
    coo_path = wt_dir / 'weight_coo.pkl'
    csr_path = wt_dir / 'weight_csr.pkl'

    data_conn = pd.read_parquet(conn_path)
    data_name = pd.read_csv(comp_path)
    num_neurons = data_name.shape[0]

    try:
        with open(coo_path, 'rb') as f:
            weight_coo = pickle.load(f)
    except FileNotFoundError:
        print('Weights not found, constructing COO weight matrix...')
        idx = [
            data_conn['Postsynaptic_Index'].to_list(),
            data_conn['Presynaptic_Index'].to_list(),
        ]
        val = data_conn['Excitatory x Connectivity'].to_list()
        weight_coo = torch.sparse_coo_tensor(
            idx, val, (num_neurons, num_neurons)
        ).to(torch.float32)
        with open(coo_path, 'wb') as f:
            pickle.dump(weight_coo, f)

    # torch.sparse.mm re-sorts an uncoalesced COO on every call, costing 44x
    # per timestep on MPS. Coalescing once moves that into setup. The
    # connectome holds no duplicate (post, pre) pairs, so nnz is unchanged and
    # the resulting matrix is bit-identical.
    weight_coo = weight_coo.coalesce()

    if csr:
        try:
            with open(csr_path, 'rb') as f:
                weight_csr = pickle.load(f)
        except FileNotFoundError:
            print('CSR weights not found, converting from COO...')
            weight_csr = weight_coo.to_sparse_csr()
            with open(csr_path, 'wb') as f:
                pickle.dump(weight_csr, f)
        return weight_csr
    else:
        return weight_coo

def silence_neurons(weights, indices):
    """Drop the outgoing synapses of the given neurons.

    This follows the reference implementation rather than the prose around it.
    code/paper-phil-drosophila/model.py silences with `syn.w['<n> == i'] = 0`,
    and Brian2's `i` on a Synapses object is the presynaptic index, so a
    silenced neuron keeps its own inputs and can still reach threshold -- its
    spikes simply arrive nowhere. The repository README describes silencing as
    cutting connections in both directions, which the code it documents does
    not do.

    The weight matrix is [post, pre], so this drops entries by column.
    """
    if not len(indices):
        return weights

    was_csr = weights.layout == torch.sparse_csr
    coo = (weights.to_sparse_coo() if was_csr else weights).coalesce()
    idx, val = coo.indices(), coo.values()

    mask = torch.zeros(coo.shape[0], dtype=torch.bool, device=idx.device)
    mask[torch.as_tensor(sorted(indices), dtype=torch.long, device=idx.device)] = True
    keep = ~mask[idx[1]]

    out = torch.sparse_coo_tensor(idx[:, keep], val[keep], coo.shape).coalesce()
    return out.to_sparse_csr() if was_csr else out


def get_nt_weights(nt_mode, conn_path, comp_path, nt_path, device_name, logger):
    """Build weights from the resolved per-neuron transmitter table.

    'signs' collapses every transmitter onto the published time constant, so
    the only change from the paper model is the polarity of neurons whose
    transmitter gates a chloride channel. 'channels' additionally gives each
    distinct time constant its own conductance.
    """
    nt_path = Path(nt_path)
    if not nt_path.exists():
        raise FileNotFoundError(
            f'{nt_path} not found; generate it with '
            'python code/prepare_neurotransmitters.py'
        )

    conn = pd.read_parquet(conn_path)
    num_neurons = pd.read_csv(comp_path).shape[0]
    nt = load_nt_table(nt_path, num_neurons)

    # Neurons absent from the connectivity table never fire into anything, so
    # their sign is irrelevant; default them to the paper's excitatory value.
    shipped = (
        conn.groupby('Presynaptic_Index')['Excitatory'].first()
        .reindex(range(num_neurons)).fillna(1).to_numpy()
    )

    force_tau = MODEL_PARAMS['tauSyn'] if nt_mode == 'signs' else None
    charge_ref = MODEL_PARAMS['tauSyn'] if nt_mode == 'channels-charge' else None
    taus, matrices, report = build_channels(
        conn, nt, num_neurons, shipped,
        force_tau=force_tau, charge_ref_tau=charge_ref,
    )

    logger.log(f"  Signs flipped:    {report['signs_overridden']} neurons")
    if len(taus) > 1 and charge_ref is None:
        # Charge per spike scales with tau here, so a shorter tau is also a
        # weaker synapse. That breaks the excitation/inhibition balance rather
        # than merely rescaling it, and no single w_syn restores the operating
        # point: MN9's rate is not monotonic in w_syn under this mode.
        logger.log(
            "  NOTE: charge per spike scales with tau, so shortening one "
            "transmitter's tau also weakens it. This mode cannot be returned "
            "to the paper's operating point by w_syn alone; use "
            "channels-charge for comparisons."
        )
    elif len(taus) > 1:
        # Charge is held fixed, but the peak conductance is not, and threshold
        # crossing depends on the peak.
        logger.log(
            "  NOTE: charge per spike is held fixed, but peak conductance is "
            "not, so w_syn still needs refitting; see code/calibrate_w_syn.py."
        )
    for ch in report['channels']:
        logger.log(
            f"  channel tau={ch['tau']:>5.1f}ms  gain={ch['gain']:>5.2f}  "
            f"neurons={ch['neurons']:>6d}  synapses={ch['synapses']:>9d}  "
            f"{', '.join(ch['classes'])}"
        )

    matrices = [m.to(device=device_name) for m in matrices]
    if nt_mode == 'signs':
        return taus, matrices[0], num_neurons
    return taus, matrices, num_neurons


# ============================================================================
# Benchmark Functions
# ============================================================================

def run_single_benchmark(t_run_sec, n_run, experiment, logger,
                         run_idx=None, total_runs=None,
                         run_label=None, round_idx=None):
    """
    Run a single PyTorch benchmark with specified t_run and n_run.

    Uses batch_size = n_run to run all trials in parallel on GPU.
    """
    device_name = resolve_device()
    nt_mode = resolve_nt_mode()
    propagation = resolve_propagation()
    compile_model = resolve_compile()
    settle_check = resolve_settle_check()
    dataset = resolve_dataset()
    ds_con, ds_comp, ds_wt = dataset_paths(dataset)
    nt_taus = None
    t_sim_ms = t_run_sec * 1000.0
    num_steps = int(t_sim_ms / DT)

    exp_name = f'pytorch_t{t_run_sec}s_n{n_run}'

    run_info = f"[{run_idx}/{total_runs}] " if run_idx else ""
    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"{run_info}BENCHMARK: t_run={t_run_sec}s, n_run={n_run}")
    logger.log_raw("=" * 80)
    logger.log(f"Device: {device_name.upper()}")
    logger.log(f"Dataset: {dataset}")
    logger.log(f"Synapse model: {nt_mode}")
    logger.log(f"Propagation: {propagation}")
    logger.log(f"torch.compile: {'on' if compile_model else 'off'}")
    logger.log(f"Settling check: {'on' if settle_check else 'off'}")
    logger.log(f"Steps: {num_steps} (dt={DT}ms)")
    logger.log(f"Experiment: {exp_name}")
    record_spikes = spike_io_enabled()
    logger.log(
        "Spike probing/output: "
        f"{'enabled' if record_spikes else 'disabled'}"
    )
    if run_label:
        logger.log(f"Run label: {run_label}")
    if round_idx is not None:
        logger.log(f"Round: {round_idx}")

    stim_rate = experiment['stim_rate']

    timings = {}
    results = {}

    try:
        # ===== Phase 1: ID mappings =====
        t_mapping_start = time()
        flyid2i, i2flyid = get_hash_tables(str(ds_comp))
        exc_indices = [flyid2i[n] for n in experiment['neu_exc']]
        slnc_ids = list(experiment.get('neu_slnc', [])) + resolve_extra_silenced()
        missing = [n for n in slnc_ids if n not in flyid2i]
        if missing:
            raise KeyError(
                f'{len(missing)} silenced id(s) are not in this connectome, '
                f'first is {missing[0]}'
            )
        slnc_indices = sorted({flyid2i[n] for n in slnc_ids})
        timings['id_mapping'] = time() - t_mapping_start
        logger.log(f"ID mapping:         {timings['id_mapping']:.3f}s")

        # ===== Phase 2: Load weights =====
        logger.log("Loading weights...")
        t_weights_start = time()
        if nt_mode == 'paper':
            # MPS implements no sparse CSR ops at all, so it takes COO.
            weights = get_weights(
                str(ds_con), str(ds_comp), str(ds_wt),
                csr=(device_name != 'mps'),
            )
            weights = weights.to(device=device_name)
            num_neurons = weights.shape[0]
        else:
            nt_taus, weights, num_neurons = get_nt_weights(
                nt_mode, str(ds_con), str(ds_comp),
                Path(path_wt) / 'neurotransmitters_783.csv',
                device_name, logger,
            )
        timings['weight_loading'] = time() - t_weights_start
        logger.log(f"  Weight loading:   {timings['weight_loading']:.3f}s")
        logger.log(f"  Neurons: {num_neurons}, Batch: {n_run}")

        if slnc_indices:
            t_slnc = time()
            if isinstance(weights, list):
                weights = [silence_neurons(w, slnc_indices) for w in weights]
                kept = sum(w._nnz() for w in weights)
            else:
                weights = silence_neurons(weights, slnc_indices)
                kept = (weights._nnz() if weights.layout == torch.sparse_coo
                        else weights.values().numel())
            logger.log(f"  Silenced:         {len(slnc_indices)} neurons, "
                       f"{kept} synapses remain ({time() - t_slnc:.3f}s)")

        # ===== Phase 3: Create model =====
        logger.log("Creating model...")
        t_model_start = time()
        if nt_mode.startswith('channels') and propagation == 'event':
            logger.log(
                "  NOTE: event propagation is implemented for the single-"
                "channel model only; using the sparse product."
            )
        if nt_mode.startswith('channels'):
            model = MultiNTTorchModel(
                n_run,
                num_neurons,
                DT,
                MODEL_PARAMS,
                nt_taus,
                weights,
                LIFNeuron(n_run, num_neurons, DT, MODEL_PARAMS, device=device_name),
                exc_indices=exc_indices,
                device=device_name
            )
        else:
            model = TorchModel(
                n_run,
                num_neurons,
                DT,
                MODEL_PARAMS,
                weights,
                exc_indices=exc_indices,
                device=device_name,
                propagate=build_propagation(propagation, weights, device_name)
            )
        conductance, delay_buffer, spikes, v, refrac = model.state_init()
        timings['model_creation'] = time() - t_model_start
        timings['model_setup_total'] = timings['weight_loading'] + timings['model_creation']
        logger.log(f"  Model creation:   {timings['model_creation']:.3f}s")
        logger.log(f"  Total setup:      {timings['model_setup_total']:.3f}s")

        mem_gb = memory_used_gb(device_name)
        if mem_gb is not None:
            logger.log(f"  Device mem after setup: {mem_gb:.2f} GB")

        # ===== Phase 4: Setup inputs =====
        rates = torch.zeros(n_run, num_neurons, device=device_name)
        rates[:, exc_indices] = stim_rate

        if compile_model:
            # Compilation happens on the first forward call. Warming it here on
            # a throwaway state keeps it out of the simulation timer, which
            # would otherwise report the compile as simulation cost.
            t_compile = time()
            model = torch.compile(model)
            warm_state = model.state_init()
            with torch.no_grad():
                for _ in range(3):
                    warm_state = model(rates, *warm_state)
            synchronize(device_name)
            timings['compile'] = time() - t_compile
            timings['model_setup_total'] += timings['compile']
            logger.log(f"  Compile:          {timings['compile']:.3f}s")
            del warm_state

        # ===== Phase 5: Run simulation =====
        logger.log(f"Running simulation ({num_steps} steps, {n_run} trial(s) batched)...")

        # Each entry is an (nnz, 3) tensor of (step, trial, neuron).
        spike_events = []
        window = spike_window_steps(n_run, num_neurons) if record_spikes else 0
        if record_spikes:
            spike_window = torch.zeros(
                window, n_run, num_neurons,
                dtype=torch.bool, device=device_name,
            )
            logger.log(f"  Spike buffer:     {window} steps on device")

        t_simulation_start = time()
        with torch.no_grad():
            for t_step in range(num_steps):
                conductance, delay_buffer, spikes, v, refrac = model(
                    rates, conductance, delay_buffer, spikes, v, refrac
                )
                if record_spikes:
                    slot = t_step % window
                    spike_window[slot] = spikes > 0
                    if slot == window - 1:
                        events = spike_window.nonzero()
                        events[:, 0] += t_step - slot
                        spike_events.append(events.cpu())

                if num_steps >= 10000 and (t_step + 1) % (num_steps // 10) == 0:
                    elapsed = time() - t_simulation_start
                    pct = (t_step + 1) / num_steps * 100
                    logger.log(
                        f"  Progress: {pct:.0f}% ({t_step+1}/{num_steps})"
                        f" - {elapsed:.1f}s elapsed"
                    )

            if record_spikes and num_steps % window:
                tail = num_steps % window
                events = spike_window[:tail].nonzero()
                events[:, 0] += num_steps - tail
                spike_events.append(events.cpu())

        synchronize(device_name)

        timings['simulation_total'] = time() - t_simulation_start
        timings['simulation_avg_per_trial'] = timings['simulation_total'] / n_run
        timings['device_build'] = 0.0
        logger.log(f"  Simulation time:  {timings['simulation_total']:.3f}s")
        logger.log(f"  Avg per trial:    {timings['simulation_avg_per_trial']:.3f}s")

        mem_gb = memory_used_gb(device_name)
        if mem_gb is not None:
            logger.log(f"  Device mem used:  {mem_gb:.2f} GB")

        # ===== Ignition check =====
        # This model has no basal firing, so with the stimulus removed any
        # activity that persists is self-sustaining rather than a response.
        # Large stimuli can latch the network into that state, where it
        # produces plenty of spikes that carry no stimulus identity -- a run
        # that looks successful and is not. Deliberately outside the timed
        # section: it must not change what the benchmark reports.
        if settle_check:
            quiet = torch.zeros_like(rates)
            windows = []
            with torch.no_grad():
                for _ in range(SETTLE_WINDOWS):
                    n = torch.zeros((), device=device_name)
                    for _ in range(int(SETTLE_WINDOW_MS / DT)):
                        conductance, delay_buffer, spikes, v, refrac = model(
                            quiet, conductance, delay_buffer, spikes, v, refrac
                        )
                        n += spikes.sum()
                    windows.append(int(n.item()))
            synchronize(device_name)

            settled = windows[-1] == 0
            results['settled'] = settled
            results['settle_windows'] = windows
            trace = " ".join(str(w) for w in windows)
            logger.log(f"  Settling ({SETTLE_WINDOWS}x{SETTLE_WINDOW_MS:.0f}ms "
                       f"quiet):  {trace}")
            if settled:
                logger.log("  Network returned to rest.")
            else:
                logger.log(
                    f"  WARNING: {windows[-1]} spikes still firing "
                    f"{SETTLE_WINDOWS * SETTLE_WINDOW_MS:.0f} ms after the "
                    f"stimulus ended. The network is self-sustaining, so this "
                    f"run's spikes do not represent a response to the stimulus."
                )

        # ===== Phase 6: Collect and save results =====
        logger.log("Collecting results...")
        t_collect_start = time()
        path_save = ''

        if not record_spikes:
            df = pd.DataFrame(
                {
                    't': [], 'time_ms': [], 'trial': [],
                    'neuron_index': [], 'flywire_id': [], 'exp_name': [],
                }
            )
            timings['result_collection'] = 0.0
            timings['result_save'] = 0.0
            logger.log("  Spike probing/output disabled; no parquet written")
        elif any(e.numel() for e in spike_events):
            events = torch.cat(spike_events)
            all_times_steps = events[:, 0].numpy()
            all_batch = events[:, 1].numpy()
            all_neurons = events[:, 2].numpy()
            all_times_ms = all_times_steps * DT

            df = pd.DataFrame({
                't': all_times_ms,
                'trial': all_batch,
                'neuron_index': all_neurons,
                'flywire_id': [i2flyid[int(n)] for n in all_neurons],
                'exp_name': exp_name,
            })
            df.insert(1, 'time_ms', df['t'])
        else:
            df = pd.DataFrame(
                {
                    't': [], 'time_ms': [], 'trial': [],
                    'neuron_index': [], 'flywire_id': [], 'exp_name': [],
                }
            )

        if record_spikes:
            timings['result_collection'] = time() - t_collect_start

            t_save_start = time()
            path_save = get_spike_output_path(exp_name, run_label, round_idx)
            path_save.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path_save, compression='brotli')
            timings['result_save'] = time() - t_save_start

        logger.log(f"  Collection:       {timings['result_collection']:.3f}s")
        logger.log(f"  Save to file:     {timings['result_save']:.3f}s")
        if path_save:
            logger.log(f"  Output file:      {path_save}")

        # ===== Calculate totals and metrics =====
        timings['total_elapsed'] = (
            timings['id_mapping']
            + timings['model_setup_total']
            + timings['simulation_total']
            + timings['result_collection']
            + timings['result_save']
        )

        total_simulated_time = t_run_sec * n_run
        timings['realtime_ratio'] = (
            total_simulated_time / timings['simulation_total']
            if timings['simulation_total'] > 0 else float('inf')
        )
        timings['realtime_ratio_total'] = (
            total_simulated_time / timings['total_elapsed']
            if timings['total_elapsed'] > 0 else float('inf')
        )

        n_active = df['flywire_id'].nunique() if len(df) > 0 else 0
        n_spikes = len(df)

        results = {
            't_run_sec': t_run_sec,
            'n_run': n_run,
            'n_active_neurons': n_active,
            'n_spikes': n_spikes,
            'status': 'success',
            'timings': timings,
            'backend_key': 'pytorch',
            'experiment_name': experiment['name'],
            'experiment_key': experiment['key'],
            'run_label': run_label or '',
            'round': round_idx or '',
            'spike_path': str(path_save),
        }

        # ===== Summary =====
        logger.log_raw("")
        logger.log_raw("-" * 60)
        logger.log("TIMING SUMMARY")
        logger.log_raw("-" * 60)
        logger.log(f"  Model setup:        {timings['model_setup_total']:>10.3f}s")
        logger.log(f"  Simulation:         {timings['simulation_total']:>10.3f}s")
        logger.log(f"  Result processing:  {timings['result_collection'] + timings['result_save']:>10.3f}s")
        logger.log(f"  -----------------------------------------")
        logger.log(f"  TOTAL ELAPSED:      {timings['total_elapsed']:>10.3f}s")
        logger.log_raw("")
        logger.log(f"  Simulated time:     {total_simulated_time:>10.1f}s ({n_run} x {t_run_sec}s)")
        logger.log(f"  Realtime ratio (sim only): {timings['realtime_ratio']:>6.3f}x")
        logger.log(f"  Realtime ratio (total):    {timings['realtime_ratio_total']:>6.3f}x")
        logger.log_raw("")
        logger.log(f"  Active neurons:     {n_active:>10d}")
        logger.log(f"  Total spikes:       {n_spikes:>10d}")
        logger.log_raw("-" * 60)

    except Exception as e:
        logger.log(f"ERROR: {str(e)}")
        logger.log_raw(traceback.format_exc())
        results = {
            't_run_sec': t_run_sec,
            'n_run': n_run,
            'n_active_neurons': 0,
            'n_spikes': 0,
            'status': f'error: {str(e)}',
            'timings': timings,
            'backend_key': 'pytorch',
            'experiment_name': experiment['name'],
            'experiment_key': experiment['key'],
            'run_label': run_label or '',
            'round': round_idx or '',
        }

    return results


def run_all_benchmarks(t_run_values=None, n_run_values=None,
                       experiment=None, logger=None,
                       run_label=None, round_idx=None):
    """
    Run all PyTorch benchmark combinations.

    Args:
        t_run_values: List of t_run durations in seconds, or None for all
        n_run_values: List of n_run values to test, or None for all
        experiment: experiment config dict from get_experiment()
        logger: BenchmarkLogger instance
    """
    if t_run_values is None:
        t_run_values = T_RUN_VALUES_SEC
    if n_run_values is None:
        n_run_values = N_RUN_VALUES
    if experiment is None:
        experiment = get_experiment()

    device_name = resolve_device()
    nt_mode = resolve_nt_mode()
    # The synapse model is part of the result's identity: a 'channels' run is
    # not comparable with the Brian2 ground truth and must not be labelled as
    # though it were.
    backend_name = f'PyTorch ({device_name.upper()})'
    if nt_mode != 'paper':
        backend_name += f' [nt:{nt_mode}]'
    if resolve_propagation() != 'sparse':
        backend_name += ' [event]'
    if resolve_compile():
        backend_name += ' [compiled]'

    benchmarks = []
    for n_run in n_run_values:
        for t_run_sec in t_run_values:
            benchmarks.append((t_run_sec, n_run))

    total_runs = len(benchmarks)

    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"BENCHMARK SUITE: {backend_name}")
    logger.log_raw("=" * 80)
    logger.log(f"Device: {device_name.upper()}")
    if device_name == 'cuda':
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
    elif device_name == 'mps':
        logger.log("GPU: Apple Metal (MPS)")
    logger.log(f"t_run values: {t_run_values} seconds")
    logger.log(f"n_run values: {n_run_values}")
    if run_label:
        logger.log(f"Run label: {run_label}")
    if round_idx is not None:
        logger.log(f"Round: {round_idx}")
    logger.log(f"Total benchmarks: {total_runs}")
    logger.log_raw("=" * 80)

    all_results = []

    for run_idx, (t_run_sec, n_run) in enumerate(benchmarks, 1):
        result = run_single_benchmark(
            t_run_sec=t_run_sec,
            n_run=n_run,
            experiment=experiment,
            logger=logger,
            run_idx=run_idx,
            total_runs=total_runs,
            run_label=run_label,
            round_idx=round_idx,
        )
        all_results.append(result)
        save_result_csv(backend_name, result)

    print_summary_table(all_results, backend_name, logger)

    return all_results
