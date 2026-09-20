# Emulation of the *Drosophila Fly* Brain

Whole-brain leaky integrate-and-fire model of the adult fruit fly, built from the
[FlyWire](https://flywire.ai/) connectome (~138k neurons, ~5M synapses).
Activate and silence arbitrary neurons; observe downstream spike propagation.

Based on the paper
[*A leaky integrate-and-fire computational model based on the connectome of the
entire adult Drosophila brain reveals insights into sensorimotor processing*](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1)
(Shiu et al.).

## Usage

With this computational model, one can manipulate the neural activity of a set of _Drosophila_ neurons.
The output of the model is the spike times and rates of all affected neurons.

Two types of manipulations are currently implemented:
- *Activation*:
Neurons can be activated at a fixed frequency to model optogenetic activation.
This triggers Poisson spiking in the target neurons. 
Two sets of neurons with distinct frequencies can be defined.
- *Silencing*:
In addition to activation, a different set of neurons can be silenced to model optogenetic silencing.
This sets that neuron's *outgoing* synaptic weights to zero: it keeps its own
inputs and can still reach threshold, but its spikes reach nothing. (An earlier
version of this sentence said connections in both directions were cut. The
reference implementation silences with `syn.w['<n> == i'] = 0`, and Brian2's `i`
is the presynaptic index, so only the outgoing side is affected.)

The entrypoint is [main.py](main.py), which parses CLI arguments and calls
[code/benchmark.py](code/benchmark.py) -- the central orchestrator that dispatches
to framework-specific runners:
[run_brian2_cuda.py](code/run_brian2_cuda.py),
[run_pytorch.py](code/run_pytorch.py),
[run_nestgpu.py](code/run_nestgpu.py), and
[run_genn.py](code/run_genn.py). The optional Brian2GeNN backend lives in
[run_brian2_genn.py](code/run_brian2_genn.py) and uses a separate conda
environment because Brian2GeNN 1.7.0 pins Brian2<2.6 while Brian2CUDA uses
Brian2 2.8.0.

```bash
# Run the 5 main-environment frameworks with default durations (0.1s–1000s)
# and trials (1,4,8,16,32)
python main.py

# Specific durations and trial count
python main.py --t_run 0.1 1 10 --n_run 1

# Single framework
python main.py --nestgpu --t_run 1 --n_run 1
python main.py --genn --t_run 1 --n_run 1
python main.py --brian2genn --t_run 1 --n_run 1

# Combine frameworks
python main.py --brian2-cpu --pytorch --t_run 0.1 1 --n_run 1 4 8 16 32

# Five-round Nature-paper benchmark suite
# Uses the March grid: t_run=(0.1,1,10,100), n_run=(1,4,8,16,32), 5 core backends
python main.py --paper --run-label nature_2026_07

# Add Brian2GeNN as the 6th framework from the brain-fly-brian2genn environment
python main.py --brian2genn --paper --run-label nature_2026_07
```

Results are incrementally saved to `data/benchmark-results.csv` as each
benchmark completes, with separate columns for setup time (loading, compilation)
and simulation time (the always-on cost). For repeated paper runs, the CSV keeps
the original March rows and appends new rows keyed by `run_label` and `round`;
the corresponding spike parquet path is recorded in `spike_path`.

Spike timing exports are written to parquet outside the timed simulation section
so file I/O does not contaminate `sim_time`. GeNN additionally flushes bounded
on-device spike-recording windows during long batched runs; that transfer time
is tracked as result collection rather than simulation time. A labeled paper run
writes partitioned outputs like:

```text
data/results/nature_2026_07/
├── manifest.csv
├── checksums.sha256
├── round_01/
│   ├── brian2cpp_t1.0s_n1.parquet
│   ├── brian2cuda_t1.0s_n1.parquet
│   ├── pytorch_t1.0s_n1.parquet
│   ├── nestgpu_t1.0s_n1.parquet
│   ├── genn_t1.0s_n1.parquet
│   └── brian2genn_t1.0s_n1.parquet
└── round_02/
```

The consolidated publication bundle contains 600 spike parquet files: 20 grid
points for each of six frameworks across five rounds. The `no_io/` subfolder
contains the corresponding one-round, 120-row timing dataset collected with
spike probing and output disabled; it intentionally contains no spike parquet
files.

Each spike parquet has one row per spike. The canonical timing column for new
exports is `time_ms`, with `trial`, `neuron_index`, `flywire_id`, and `exp_name`.
The legacy `t` column is kept for existing analysis scripts.

The full `nature_2026_07` spike parquet bundle is too large for regular Git
tracking, so parquet files are intentionally gitignored. The committed metadata
files are `manifest.csv` and `checksums.sha256`; the full bundle is stored in
Google Drive:

https://drive.google.com/drive/folders/1jiSfb5lNfm9gwP0YyyRz5ATIrDpBAcjs

After downloading the Drive folder into `data/results/nature_2026_07/`, verify
the bundle with:

```bash
cd data/results/nature_2026_07
sha256sum -c checksums.sha256
```

### Apple Silicon (MPS)

The PyTorch backend runs on Apple's Metal backend. Device selection is automatic
(`cuda` > `mps` > `cpu`) and can be pinned with `FLYBRAIN_TORCH_DEVICE`, which
fails loudly rather than demoting to CPU if the named backend is unavailable.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-mps.txt

.venv/bin/python main.py --pytorch --t_run 0.1 --n_run 1

# pin a device, e.g. to compare MPS against CPU
FLYBRAIN_TORCH_DEVICE=cpu .venv/bin/python main.py --pytorch --t_run 0.1 --n_run 1
```

`environment.yml` does not resolve on macOS -- it pins brian2cuda, GeNN and the
CUDA wheel index -- so use `requirements-mps.txt`. Without an NVIDIA GPU only
`--brian2-cpu` and `--pytorch` are available.

MPS implements no sparse CSR operations, so the weight matrix is held in COO on
that device. Two further details are what make the sparse kernel actually
engage: the recurrent step is written as `torch.sparse.mm(W, spikes.T).T`
rather than `matmul(spikes, W.T)`, and the COO matrix is coalesced once at load
time. Skipping either drops onto a fallback path costing roughly 190 ms per
timestep. Measured on an M3 (138,639 neurons, 15.1M synapses, `t_run=0.1`,
`n_run=1`):

| path | simulation time |
| --- | --- |
| CPU | 51.3 s |
| MPS | 5.7 s |

With the Poisson input held fixed, spike trains and membrane voltages are
bit-identical across CPU/CSR, CPU/COO and MPS/COO, and unchanged from the
`matmul(spikes, W.T)` formulation this replaced.

### Settling check

This model has two regimes and only one of them is a response. Drive a small
set of neurons and activity rises, spreads and stops: the published sugar
protocol reaches 365 neurons and is silent again 50 ms after the stimulus ends.
Drive a larger set -- around 200 receptor neurons at 20 Hz, or 200 random
neurons at 100 Hz -- and the network latches into a self-sustaining state that
never returns to rest. It still produces spikes, a great many of them, but they
carry no stimulus identity: two different odours in that state recruit the same
cells, Jaccard 0.99.

Nothing distinguished the two from the outside. A latched run completes,
reports `success`, and writes a plausible-looking raster. So every run now ends
with a short quiet period and reports whether the network returned to rest.
The model has no basal firing, so the criterion needs no threshold: with the
stimulus removed, any activity at all is self-sustaining.

```
Settling (4x25ms quiet):  190 65 0 0
Network returned to rest.
```

```
Settling (4x25ms quiet):  11556 11649 11197 11833
WARNING: 11833 spikes still firing 100 ms after the stimulus ended. The
network is self-sustaining, so this run's spikes do not represent a response
to the stimulus.
```

The check runs outside the timed section, so `sim_time` and the other
benchmark figures are unchanged; only wall-clock grows, by the length of the
quiet period. `FLYBRAIN_SETTLE_CHECK=0` turns it off. Both shipped
experiments settle: sugar GRNs at 200 Hz and P9 at 100 Hz.

### Lesions and controls

Silencing is how a control is run against this model, and until now the PyTorch
backend ignored the `neu_slnc` field in an experiment definition entirely: a
lesion study configured there would have produced intact results with no error
and no warning. The backend now honours it, and `FLYBRAIN_SILENCE` adds ids on
top so a control does not require editing `benchmark.py`:

```bash
FLYBRAIN_SILENCE=720575940624963786,720575940630233916 \
  .venv/bin/python main.py --pytorch --t_run 0.1 --n_run 1
```

Ids not present in the connectome raise rather than being skipped. Silencing all
21 sugar GRNs removes 1,550 outgoing synapses and takes the response from 348
active neurons to 21 — the GRNs keep firing on their Poisson drive, and nothing
downstream of them does.

### MaleCNS: one animal, whole nervous system

Joining FlyWire to MANC produces a chimera, and only 39% of the descending
neurons could be matched. MaleCNS avoids both: one male fly's entire central
nervous system, brain and nerve cord imaged as a single volume, so descending
neurons arrive with both halves already attached and nothing has to be sewn.

```bash
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
mkdir -p data/malecns && cd data/malecns
curl -O $B/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl -O $B/body-neurotransmitters-male-cns-v1.0.feather
curl -O $B/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather
cd ../.. && .venv/bin/python code/build_malecns.py

FLYBRAIN_DATASET=malecns FLYBRAIN_W_SYN=0.10 \
  .venv/bin/python main.py --pytorch --t_run 0.1 --n_run 1
```

| | brain | cns | malecns |
| --- | --- | --- | --- |
| neurons | 138,639 | 161,291 | 165,122 |
| connections | 15.1 M | 20.3 M | 25.6 M |
| synapses | 54.5 M | 85.2 M | **124.0 M** |
| motor neurons | 110 | 434 | 815 |
| one animal | brain only | no | **yes** |

It also ships its own transmitter predictions, 83,496 of them backed by a
literature ground truth, and they name histamine. The correction that
`prepare_neurotransmitters.py` recovers for FlyWire -- 7,362 photoreceptors
scored excitatory because the six-class classifier cannot emit histamine --
simply does not arise here: 5,910 neurons come labelled histaminergic and
inhibitory.

**It needs its own w_syn.** That parameter is a fit, not a property of the
equations, and it was fitted to FlyWire. At the published 0.275 the sugar
protocol drives MaleCNS into the self-sustaining state, 14,501 neurons firing
and still firing 100 ms after the stimulus stops. Around 0.10 it settles: 248
neurons, quiet again within 25 ms, which is the scale of FlyWire's own
response. `FLYBRAIN_W_SYN` sets it. Settling is a bound, not a calibration --
fitting it properly needs an observable, and `code/calibrate_w_syn.py` reads
out MN9, which is a FlyWire cell.

Experiments are defined by FlyWire root id, so on this dataset they are carried
across by cell type and the runner reports what that resolved to. It is not
always the same stimulus: the sugar experiment names 21 right-hemisphere LB3
neurons and this volume holds 87 of them.

### The ventral nerve cord

FlyWire is a brain. It stops at the neck, so a descending neuron in it is half
a cell: the model can excite one and watch it fire, and the spikes go nowhere,
because the motor circuits they command are in the ventral nerve cord and the
nerve cord is not in the dataset.

`code/build_cns.py` sews the two halves back together, using the MANC nerve
cord connectome (CC-BY, Janelia) and the descending neuron correspondence from
the neck connective work of Stürner et al., whose supplemental tables carry a
FlyWire root id and a MANC body id under one harmonised type vocabulary.
Matching on (type, side) joins 394 groups: 513 brain descending neurons, 39.1%
of them, are reunited with their axons.

```bash
.venv/bin/python code/build_cns.py                       # builds it once
FLYBRAIN_DATASET=cns .venv/bin/python main.py --pytorch --t_run 0.1 --n_run 1
```

| | brain | cns |
| --- | --- | --- |
| neurons | 138,639 | 161,291 |
| connections | 15.1 M | 20.3 M |
| synapses | 54.5 M | 85.2 M |
| motor neurons | 110 | 434 |

Descending commands now arrive somewhere. Driving DNa02, a steering neuron,
recruits 611 nerve cord neurons and fires 33 motor neurons; MDN, which drives
backward walking, fires 27. Both settle afterwards. The sugar protocol gains
about 190 nerve cord neurons over the brain alone.

Three things to keep in mind. FlyWire is a female brain and MANC a male nerve
cord, so this animal is a chimera. The 61% of descending neurons that could not
be matched keep both halves and no join between them, so any command routed
through those cells still stops at the neck. And 55 of the 536 sewn pairs
disagree between the two datasets about their own transmitter; the brain's
assignment is kept and the count is printed, not hidden.

The per-transmitter synapse models (`FLYBRAIN_NT_MODE`) cover the brain only
and refuse to run on this dataset rather than silently mis-signing the cord.

### Neurotransmitter identity and synaptic kinetics

The shipped connectivity parquet stores transmitter identity as a single
`Excitatory` column of +1/-1, assigned per neuron by the majority rule in Shiu
et al. That collapse loses two things. Monoamine neurons are neither GABAergic
nor glutamatergic, so they score +1 and become indistinguishable from
cholinergic neurons. And the Eckstein et al. (2024) classifier predicts six
transmitters and cannot emit histamine at all, so histaminergic neurons --
photoreceptors, whose output gates Ort/hclA chloride channels and is therefore
inhibitory -- also score +1.

`code/prepare_neurotransmitters.py` recovers the identity from the
[FlyWire annotations](https://github.com/flyconnectome/flywire_annotations) and
writes `data/neurotransmitters_783.csv`. Every neuron carries its provenance:

```bash
.venv/bin/python code/prepare_neurotransmitters.py
```

| provenance | neurons | share |
| --- | --- | --- |
| literature (`known_nt`) | 76,741 | 55.35% |
| classifier (`top_nt`) | 61,754 | 44.54% |
| none | 144 | 0.10% |

`FLYBRAIN_NT_MODE` then selects the synapse model:

| mode | behaviour |
| --- | --- |
| `paper` (default) | the published model, bit-identical to before |
| `signs` | published model, transmitter signs corrected |
| `channels` | one alpha conductance per distinct time constant |

```bash
FLYBRAIN_NT_MODE=signs .venv/bin/python main.py --pytorch --t_run 0.1 --n_run 1
```

Correcting the signs flips 12,233 neurons, 7,362 of them histaminergic. The
weight-level effect is much smaller than the neuron-level one: those neurons sit
upstream of 1.81% of all synapses, because photoreceptor arbours are only
partly reconstructed in FAFB.

**The time constants are mostly not constrained.** Only acetylcholine has a
usable measurement (2 ms decay, Lee & O'Dowd 1999). GABA has separate fast
GABA-A and slow GABA-B components whose balance depends on postsynaptic
receptor expression, which the connectome does not carry; for glutamate no
central-synapse decay constant was found, and the figures in circulation are
oocyte desensitisation constants, a different quantity. Those entries stay at
the published 5 ms and are marked `UNCONSTRAINED` in `NT_KINETICS`
(`code/model_multi_nt.py`), with the reasoning recorded per entry. Nothing is
filled in with a plausible-looking guess.

One consequence to keep in mind when reading `channels` output: `w_syn` was
fitted with a single 5 ms constant, and in this alpha synapse a pulse transfers
charge proportional to tau. Shortening a transmitter's tau also weakens it, so
firing rates move for reasons other than kinetics until `w_syn` is refitted.
The runner prints this warning whenever more than one channel is active.

A `channels` run is not comparable with the Brian2 ground truth and is labelled
`PyTorch (MPS) [nt:channels]` in the results CSV so it cannot be mistaken for
one.

### Spike propagation

The published formulation multiplies the whole connectome every timestep,
which costs the same whether the brain is silent or saturated: 4.38 ms on MPS
with no neuron firing, 4.19 ms with 5000 firing. Under the stimulus protocols
here about 1.7 neurons fire per step, so nearly all of that multiplies zeros.

`FLYBRAIN_PROPAGATION=event` gathers only the outgoing synapses of neurons
that fired instead. Output is bit-identical -- 166M spike entries compared on
both CPU and MPS with zero mismatches -- so the choice is purely one of cost.

End-to-end ms/step, 300 steps, spike recording off, median of 3 paired repeats:

| device | batch | `sparse` | `event` | speedup |
| --- | --- | --- | --- | --- |
| MPS | 1 | 9.36 | **2.13** | 4.4x |
| MPS | 8 | 34.49 | 17.75 | 1.9x |
| MPS | 32 | 112.13 | 60.38 | 1.9x |
| CPU | 1 | 84.73 | **1.81** | 46.7x |
| CPU | 8 | 106.05 | 16.20 | 6.5x |
| CPU | 32 | 213.49 | 76.55 | 2.8x |

Two consequences worth knowing before choosing a configuration.

**CPU beats MPS at small batch once propagation is event-driven.** The device
split the step between them: MPS runs the dense state update 5.8x faster
(0.26 ms against 1.52 ms) while CPU runs the event gather 6.4x faster (0.29 ms
against 1.87 ms), because MPS spends most of an event step in kernel launch
and one synchronisation regardless of how few spikes there are. Those cancel,
and CPU comes out ahead until batch 32.

**Batching stops paying on CPU and MPS.** With `sparse`, per-trial cost falls
from 9.36 to 3.50 ms between batch 1 and 32 on MPS. With `event` it is flat,
1.8-2.4 ms at every batch on both, because event work scales with the batch and
no fixed cost remains to amortise. Multi-trial runs are better as separate
batch-1 processes there. CUDA behaves differently and is covered below.

**On CUDA the window where `event` pays is narrow.** Measured on a GTX 1660
SUPER, the sparse product costs a flat 1.48 ms per step whatever the activity,
which is 7x cheaper than MPS and 55x cheaper than CPU. `event` beats it only in
the very sparse regime the stimulus protocols produce:

| | batch 1 | batch 8 | batch 32 |
| --- | --- | --- | --- |
| `sparse` | 1.49 ms | 11.81 ms | 19.22 ms |
| `event` | 1.20 ms | 3.14 ms | 8.92 ms |
| per trial (`event`) | 1.196 ms | 0.392 ms | **0.279 ms** |

Drive every neuron instead and `event` loses at every level tested, from 0.86x
at 160 spikes per step down to 0.16x at 27,721. Unlike CPU and MPS, batching
does pay here: per-trial cost falls 4.3x between batch 1 and 32, making CUDA at
batch 32 the cheapest per-trial configuration measured anywhere.

Raise both at once and `event` does not merely lose, it collapses: at batch 32
with every neuron driven to 1000 Hz it takes 2155 ms per step against 19 ms for
the sparse product, 112x worse, and peaks at 5.02 of 6 GB of VRAM. There are
then 13,890 spikes per trial across 32 trials, so the gather touches about 48M
synapses per step, three times the 15.1M in the whole connectome. On CUDA,
`sparse` is the right default and `event` is for sparse protocols only -- the
opposite of the rule on Apple Silicon.

The rule across devices is that the faster a device runs the sparse product,
the narrower the activity range in which `event` is worth using: CPU wins
everywhere tested, MPS below roughly 100 Hz mean firing, CUDA only in sparse
protocols.

`torch.compile` gains nothing on CUDA (1.425 against 1.428 ms), unlike the 28%
it gives on CPU.

Every CUDA configuration was checked against the CPU reference and is
bit-identical: 83M spike entries, zero mismatches, zero difference in final
membrane voltage, with and without `event` and with and without compilation.

Event propagation is implemented for the single-channel model. The per-channel
models (`channels`, `channels-charge`) keep the sparse product and say so.

`FLYBRAIN_COMPILE=1` additionally runs the model through `torch.compile`.
Output stays bit-identical (83M spike entries, zero mismatches, on every
configuration tested), so this too only trades a one-off compilation for a
cheaper step: 1.98 to 1.43 ms on CPU with event propagation (t=6.3 over ten
interleaved pairs), 3.80 to 3.39 ms on MPS, and 1% on the sparse product. The
data-dependent gather length in `propagation.py` breaks the graph, but the
dense state update still fuses around it. Compilation is warmed on a throwaway
state before the timer starts, and costs 0.2-2.9 s once.

Spike recording is buffered on the device and read back once per window rather
than every step, which removes a per-step synchronisation worth 0.282 ms
(sd 0.085, ten interleaved pairs). Timings on this machine drift by more than
that between runs, so comparisons have to be paired.

### Calibrating w_syn

`w_syn` is the model's only free parameter; every other constant is cited to a
measurement. Anything that changes synaptic gain invalidates it, so
`code/calibrate_w_syn.py` refits it against the sugar-GRN/MN9 protocol.

```bash
.venv/bin/python code/calibrate_w_syn.py --nt-mode paper --dose-response
.venv/bin/python code/calibrate_w_syn.py --nt-mode paper --noise 5
.venv/bin/python code/calibrate_w_syn.py --nt-mode signs
```

**The published criterion does not reproduce here.** Shiu et al. chose `w_syn`
so that 100 Hz sugar-GRN drive puts MN9 at roughly 80% of maximal firing,
without defining maximal, and neither reading of it survives contact with the
v783 data.

Read as a plateau in the dose-response curve, there is no plateau — MN9 is
still rising at 600 Hz, and its 100 Hz rate is 52% of the 600 Hz rate and 62%
of the rate at 200 Hz, the top of the paper's own sweep:

| sugar GRN drive | 25 Hz | 50 | 75 | 100 | 200 | 400 | 600 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MN9 | 0.0 | 17.9 | 57.9 | 63.3 | 101.7 | 117.9 | 121.2 |

Read as a ceiling over `w_syn`, the quantity does not exist. MN9's rate is not
monotonic in `w_syn`, because raising it amplifies the 30% of neurons that are
inhibitory just as much: `w_syn=5.0` gives a *lower* MN9 rate than the
published 0.275, and `w_syn=3.125` silences MN9 outright. The paper also
calibrated against FlyWire v630 while this repository ships v783, and the two
sugar GRN sets differ by one root id.

The script therefore fits `w_syn` so that each synapse model reaches the same
MN9 rate as the published model does at its published `w_syn`. That is an
operational substitute, not the paper's criterion, and it is labelled as such.

**Read any fit against the noise floor first.** The Poisson input is unseeded,
so repeating one configuration scatters: 69.9 Hz mean, 1.4 Hz sd over five
repeats, and 64.8-71.7 Hz across separate invocations. The script averages the
reference over `--reference-repeats` and says so explicitly when a fitted shift
is no larger than that scatter.

On that basis, `signs` needs no measurable change: the correction moves only
1.81% of synaptic weight and the fitted shift sits inside the noise. Keep the
published 0.275.

**Raw `channels` cannot be calibrated by `w_syn` at all.** Between 0.44 and
4.61 the MN9 rate runs 0.0, 11.7, 14.6, 4.0, 6.2, 64.2 Hz, and the script
refuses to bisect through that rather than report a fit from it. The cause is
mechanistic, not numerical: shortening only the cholinergic time constant
leaves inhibition transferring 2.5x the charge per spike, so the
excitation/inhibition balance is broken and no single scalar restores it.

`channels-charge` exists for that reason. It scales each channel by
`tau_ref/tau`, holding the charge per presynaptic spike fixed so that only the
time course of the conductance differs. It is the mode to use when comparing
kinetics against the published model; raw `channels` reflects the parameter
table as written and is left available deliberately.

Holding charge fixed does not by itself restore the operating point, because
threshold crossing follows the peak conductance and not the integral: at the
published `w_syn`, `channels-charge` drives MN9 to 129.8 Hz against the paper
model's 69.0 Hz. It does restore monotonicity, so the fit converges.

| mode | w_syn | MN9 at the published w_syn |
| --- | --- | --- |
| `paper` | 0.275 | 69.0 Hz (reference, mean of 3) |
| `signs` | 0.275 — unchanged | within the reference scatter |
| `channels` | not fittable | — |
| `channels-charge` | **0.1977** (ratio 0.719) | 129.8 Hz |

The `signs` row is a measurement, not an omission: two independent fits landed
at 0.2653 and 0.2802, straddling the published 0.275, which is what noise looks
like rather than a shift.

### Ground truth comparison

Brian2 (CPU) serves as the ground truth for neural accuracy: it implements the
canonical LIF model from
[Shiu et al. (Nature 2024)](https://www.nature.com/articles/s41586-024-07763-9),
which achieved 91% prediction accuracy against experimental _Drosophila_ data.
Each backend also saves per-neuron spike trains to `data/results/`, and a
comparison script measures how closely the other backends reproduce Brian2's
output:

```bash
python code/compare_ground_truth.py                  # default: t_run=1s, n_run=1
python code/compare_ground_truth.py --t_run 10 --n_run 4   # longer / averaged
python code/compare_ground_truth.py --run-label nature_2026_07 --round 1
```

This computes active-neuron overlap (Jaccard), per-neuron firing-rate
correlation, and spike-count ratios, and writes structured results to
`data/ground-truth-comparison.json`.

For all-framework pairwise comparisons, including firing-rate parity rows and
spike-time matches within a tolerance window, use:

```bash
python code/compare_spike_outputs.py \
  --run-label nature_2026_07 \
  --round 1 \
  --output-dir data/results/nature_2026_07/comparisons
```

This writes `pairwise_summary.csv`, `pairwise_summary.json`,
`parity_rates.csv`, and `missing_inputs.json`. The pairwise summary has one row
per framework pair and `t_run`/`n_run` combination.

For paper-support parity files comparing one backend against Brian2 CPU across
all five labeled rounds, use:

```bash
python code/compare_backend_to_brian2.py \
  --run-label nature_2026_07 \
  --backend brian2genn \
  --output-dir data/results/nature_2026_07/comparisons
```

This writes `<backend>_vs_brian2_rate_summary.csv/json`,
`<backend>_vs_brian2_rate_parity.csv`, and
`<backend>_vs_brian2_missing_inputs.json`. Add `--include-timing` only for
smaller targeted checks where greedy spike-time matching is scientifically
useful and computationally reasonable.

## Installation

### Conda environment

The `brain-fly` conda environment provides everything needed to run the
**Brian2**, **Brian2CUDA**, **PyTorch**, **NEST GPU**, and **GeNN** backends
(including CUDA-enabled PyTorch and PyGeNN):

```bash
conda env create -f environment.yml
conda activate brain-fly
```

On Ubuntu/WSL, PyGeNN's source build also needs the system `pkg-config` binary
and libffi headers:

```bash
sudo apt-get install -y pkg-config libffi-dev
```

### GeNN

The `--genn` backend uses PyGeNN 5.4.0 with the CUDA backend. It implements
Brian2-style Poisson activation into membrane voltage, delayed sparse recurrent
synapses, GeNN batching for `n_run`, and the same parquet spike schema as the
other benchmark runners.

Large batched GeNN runs cap the on-device spike recording buffer with
`GENN_RECORDING_WINDOW_MAX_SLOTS` (default: `800000`). This preserves full spike
timing exports while avoiding CUDA out-of-memory errors for large
`n_run * t_run` combinations.

If PyGeNN was not installed when the conda environment was created, install it
inside `brain-fly` with:

```bash
export CUDA_PATH=/usr/local/cuda-12.5
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_PATH/bin:$PATH
pip install https://github.com/genn-team/genn/archive/refs/tags/5.4.0.zip
```

### Brian2GeNN

The `--brian2genn` backend uses Brian2GeNN 1.7.0 as a Brian2 standalone device
targeting GeNN/CUDA. It is intentionally isolated from the main `brain-fly`
environment because Brian2GeNN pins Brian2<2.6, which conflicts with
Brian2CUDA's Brian2 2.8.0 requirement.

Create the environment with:

```bash
conda env create -f environment-brian2genn.yml
conda activate brain-fly-brian2genn
```

Brian2GeNN 1.7.0 expects GeNN 4.x command-line scripts such as
`genn-buildmodel.sh`. If they are not already installed, place GeNN 4.9.0 at
`~/.local/src/genn-4.9.0` or set `BRIAN2GENN_GENN_PATH`/`GENN_PATH` to your
GeNN 4.x source tree:

```bash
export CUDA_PATH=/usr/local/cuda-12.5
export CUDA_HOME=$CUDA_PATH
export BRIAN2GENN_GENN_PATH=$HOME/.local/src/genn-4.9.0
export GENN_PATH=$BRIAN2GENN_GENN_PATH
export PATH=$GENN_PATH/bin:$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

For scientific comparability, the Brian2GeNN runner exports the same per-spike
parquet schema as the other backends and uses the same upstream Poisson drive.
Brian2GeNN cannot run this model's independent trials as a true GeNN batch in
the way the direct `--genn` backend can, so `n_run>1` is implemented as
independent build/run trials with deterministic per-trial C RNG seeds. The
`sim_time` column records GeNN executable time; `build_time` records the
Brian2GeNN code generation/compilation overhead.

### NEST GPU

NEST GPU requires a separate build from source with a custom neuron model
(`user_m1`). This is only needed if you want to use the `--nestgpu` backend.

**Prerequisites:**

- **NVIDIA CUDA Toolkit** (12.x) — follow the
  [official installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
- **CMake** — `sudo apt install cmake` (or see
  [cmake.org](https://cmake.org/download/)).

**Steps:**

1. Clone NEST GPU:

```bash
git clone https://github.com/nest/nest-gpu
```

2. Copy the custom source files into the NEST GPU tree. You must replace `/path/to/nest-gpu` with your own local path:

```bash
cp scripts/nestgpu_source_files/src/user_m1.{h,cu}    /path/to/nest-gpu/src/
cp scripts/nestgpu_source_files/pythonlib/nestgpu.py   /path/to/nest-gpu/pythonlib/
```

   The patched `nestgpu.py` fixes weight array initialization (lines 2225-2227).

3. Build and install (set `-DCMAKE_CUDA_ARCHITECTURES` to match your GPU, e.g.
   `89` for RTX 4070):

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_INSTALL_PREFIX=$HOME/.nest-gpu-build \
      /path/to/nest-gpu
make -j$(nproc) && make install
```

For a full setup from a fresh Windows machine (WSL2 + CUDA + Miniconda), see
[scripts/setup_WSL_CUDA.sh](scripts/setup_WSL_CUDA.sh).

----

## Frameworks

| Framework | Backend | Status |
|---|---|---|
| **Brian2** | C++ standalone (multi-core CPU) | ready |
| **Brian2CUDA** | CUDA standalone (GPU) | ready |
| **PyTorch** | CUDA (GPU) | ready |
| **NEST GPU** | CUDA (GPU, custom `user_m1` neuron) | ready |
| **GeNN** | CUDA (GPU, PyGeNN 5.4.0) | ready |
| **Brian2GeNN** | Brian2GeNN 1.7.0 / GeNN CUDA | ready, separate env |

All six frameworks share the same data, model parameters, spike-output schema,
and folder structure. The five main backends run from `brain-fly` plus a
system-level NEST GPU install; Brian2GeNN runs from `brain-fly-brian2genn`
because of its Brian2 version pin.

## Quickstart

```bash
# Create the conda environment (includes CUDA-enabled PyTorch)
conda env create -f environment.yml
conda activate brain-fly

# Run a 1-second benchmark on the five main-environment backends
python main.py --t_run 1 --n_run 1 --no_log_file

# Specific backends (combinable)
python main.py --brian2-cpu                    # Brian2 CPU only
python main.py --brian2cuda-gpu               # Brian2CUDA GPU only
python main.py --pytorch                      # PyTorch only
python main.py --nestgpu                      # NEST GPU only
python main.py --genn                         # GeNN only
python main.py --brian2genn                   # Brian2GeNN only, from brain-fly-brian2genn
python main.py --pytorch --genn               # PyTorch + GeNN

# Full benchmark suite (all durations, n_run=1,4,8,16,32, five main backends)
python main.py

# Nature-paper suite: five main backends, March parameter grid, 5 rounds
python main.py --paper --run-label nature_2026_07

# Brian2GeNN Nature-paper add-on from the separate brain-fly-brian2genn env
python main.py --brian2genn --paper --run-label nature_2026_07
```

### `main.py` options

| Flag | Description |
|---|---|
| *(default)* | Run all: Brian2 (CPU) → Brian2CUDA (GPU) → PyTorch → NEST GPU → GeNN |
| `--brian2-cpu` | Brian2 C++ standalone (CPU) only |
| `--brian2cuda-gpu` | Brian2CUDA (GPU) only |
| `--pytorch` | PyTorch (GPU/CPU) only |
| `--nestgpu` | NEST GPU only |
| `--genn` | GeNN CUDA backend only |
| `--brian2genn` | Brian2GeNN backend only; use the `brain-fly-brian2genn` environment |
| `--t_run` | Simulation duration(s) in seconds, e.g. `--t_run 0.1 1 10` |
| `--n_run` | Number of independent trials, e.g. `--n_run 1 4 8 16 32` |
| `--paper` | Run the paper suite: `t_run=[0.1,1,10,100]`, `n_run=[1,4,8,16,32]`, 5 rounds |
| `--rounds` | Repeat the full selected backend/parameter suite N times |
| `--round-start` | First round number to write, useful for resuming a labeled run |
| `--run-label` | Group repeated spike outputs under `data/results/<label>/` and append labeled CSV rows |
| `--log_file FILE` | Write log to file (default: `data/results/benchmarks.log`) |
| `--no_log_file` | Console output only |

Backend flags are combinable: `--brian2-cpu --pytorch` runs Brian2 CPU then PyTorch.

## Project structure

```
fly-brain/
├── main.py                     # Entrypoint (benchmark runner CLI)
├── environment.yml             # Conda env definition (brain-fly)
├── environment-brian2genn.yml  # Separate Brian2GeNN env definition
├── code/
│   ├── benchmark.py            # Orchestrator: config, logging, dispatcher
│   ├── run_brian2_cuda.py      # Brian2 / Brian2CUDA benchmark runner
│   ├── run_pytorch.py          # PyTorch benchmark runner (model + utils)
│   ├── run_nestgpu.py          # NEST GPU benchmark runner (subprocess per trial)
│   ├── run_genn.py             # GeNN/PyGeNN benchmark runner
│   ├── compare_ground_truth.py # Compare backends against Brian2 (CPU) ground truth
│   └── paper-brian2/           # Original paper code (not used by benchmarks)
│       ├── model.py            # Core LIF network model (Brian2)
│       ├── utils.py            # Analysis helpers (load_exps, get_rate)
│       ├── example.ipynb       # Tutorial: activation, silencing, rate analysis
│       └── figures.ipynb       # Reproduce paper figures (uses archive 630 data)
├── data/
│   ├── 2025_Completeness_783.csv       # Neuron list (FlyWire v783)
│   ├── 2025_Connectivity_783.parquet   # Synapse connectivity (FlyWire v783)
│   ├── benchmark-results.csv           # Accumulated benchmark timings
│   ├── ground-truth-comparison.json   # Backend accuracy vs Brian2 (CPU)
│   ├── sez_neurons.pickle              # SEZ neuron subset (for figures)
│   ├── weight_coo.pkl                  # Cached sparse weights COO (gitignored)
│   ├── weight_csr.pkl                  # Cached sparse weights CSR (gitignored)
│   ├── archive/
│   │   ├── 2023_Completeness_630.csv   # Legacy v630 data
│   │   └── 2023_Connectivity_630.parquet
└── scripts/
    └── setup_WSL_CUDA.sh       # WSL2 + CUDA + Miniconda setup
```

## Data

The model uses FlyWire connectome data version **783** (public release).
Legacy version 630 data is kept in `data/archive/` for paper figure reproduction.

| File | Description | Size |
|---|---|---|
| `2025_Completeness_783.csv` | Neuron IDs and metadata | 3.2 MB |
| `2025_Connectivity_783.parquet` | Pre/post-synaptic indices + weights | 97 MB |
| `weight_coo.pkl` | Sparse weight matrix (COO), auto-generated by PyTorch | ~288 MB |
| `weight_csr.pkl` | Sparse weight matrix (CSR), auto-generated by PyTorch | ~289 MB |

## Architecture per framework

| | Brian2 / Brian2CUDA | PyTorch | NEST GPU |
|---|---|---|---|
| Build step | C++ / CUDA codegen + compile | None (eager mode) | None |
| Trial parallelism | Sequential (`device.run`) | Batched (`batch_size=n_run`) | Subprocess per trial (cannot reset in-process) |
| Weight format | Brian2 `Synapses` object | Sparse CSR tensor | Array-based `Connect` |
| Neuron model | Brian2 equations | Custom `nn.Module` classes | Custom CUDA kernel (`user_m1`) |
| Timestep | 0.1 ms | 0.1 ms | 0.1 ms |

## System requirements

- Linux (tested on Ubuntu 22.04 under WSL2 on Windows 11)
- NVIDIA GPU with CUDA 12.x (tested on RTX 4070)
- Miniconda / Anaconda
- NEST GPU compiled from source (for `--nestgpu` backend)
- `scripts/setup_WSL_CUDA.sh` documents the full setup from a fresh Windows machine

## License

Except where otherwise noted, this project is licensed under the GNU General
Public License version 2 or any later version
(`GPL-2.0-or-later`). See [LICENSE](LICENSE).

Third-party components retain their original notices. In particular, the
Shiu et al. Brian2 materials in `code/paper-phil-drosophila/` remain available
under their upstream [MIT License](code/paper-phil-drosophila/LICENSE), and the
adapted NEST GPU model files retain their GPL-2.0-or-later notices.
