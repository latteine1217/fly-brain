"""
Refit w_syn so that the synapse models share one operating point.

w_syn is the model's only free parameter; every other constant is cited to a
measurement. Anything that changes synaptic gain therefore invalidates it.
Correcting transmitter signs changes gain, and giving a transmitter its own
time constant changes it more: in this alpha synapse a presynaptic pulse
transfers charge proportional to tau, so shortening tau also weakens the
synapse. Comparing firing rates across FLYBRAIN_NT_MODE settings without
refitting measures the drift in w_syn as much as the mechanism.

On the published criterion
--------------------------
Shiu et al. state that w_syn was chosen so that driving the labellar sugar GRNs
at 100 Hz puts MN9 at roughly 80% of maximal firing, without defining maximal.
That criterion does not reproduce here, and the sweeps below are why:

  * Read as a plateau in the dose-response curve, there is no plateau. MN9
    rises monotonically with sugar frequency and is still rising at 600 Hz. At
    the published w_syn, its 100 Hz rate is 52% of the 600 Hz rate and 62% of
    the rate at 200 Hz, the top of the paper's own sweep.

  * Read as a ceiling over w_syn, the quantity does not exist. MN9's rate is
    not monotonic in w_syn, because raising it amplifies the 30% of neurons
    that are inhibitory just as much: w_syn=5.0 yields a lower MN9 rate than
    the published 0.275, and w_syn=3.125 silences MN9 entirely.

The paper also calibrated against FlyWire v630, while this repository ships
v783, and the two sugar GRN sets differ by one root id.

So this script does not try to recover the published target. It fits w_syn so
that each synapse model reaches the same MN9 rate as the published model does
at its published w_syn, under the same stimulus. That is an operational
substitute, chosen because it serves what recalibration is for -- holding the
operating point fixed so that a difference between modes is attributable to the
mechanism rather than to gain.

What each mode needs
--------------------
signs            no measurable change; the correction moves 1.81% of synaptic
                 weight and the fitted shift falls inside the noise.
channels         not fittable. MN9's rate is not monotonic in w_syn, because
                 shortening only the cholinergic tau leaves inhibition
                 transferring 2.5x the charge per spike. The bracketing step
                 detects this and refuses rather than bisecting through it.
channels-charge  fittable, and the shift is real rather than noise.

Read any fit against the noise floor. The Poisson input is unseeded, so one
configuration repeated scatters by about 2% sd within a run and more across
runs; --noise measures it and --reference-repeats averages the reference.

Usage:
    python code/calibrate_w_syn.py --nt-mode paper --dose-response
    python code/calibrate_w_syn.py --nt-mode paper --noise 5
    python code/calibrate_w_syn.py --nt-mode channels-charge
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow  # noqa: F401  -- import before torch, see run_pytorch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import get_experiment, path_comp, path_con, path_wt  # noqa: E402
from model_multi_nt import MultiNTTorchModel, build_channels, load_nt_table  # noqa: E402
from run_pytorch import (  # noqa: E402
    DT, MODEL_PARAMS, LIFNeuron, TorchModel, get_hash_tables, get_weights,
    resolve_device, synchronize,
)

# Proboscis motor neuron, left. Taken from the notebooks in
# code/paper-phil-drosophila, where it is the readout for figures 1E and 3A.
MN9_LEFT = 720575940660219265

# The stimulus the published calibration used.
CALIBRATION_HZ = 100.0

# Synaptic drive needs a few membrane time constants to settle. MN9's rate is
# stationary after this point: measured in 100 ms bins over a 1 s trial it
# fluctuates between 60 and 77 Hz with no trend.
SETTLE_MS = 100.0

DOSE_RESPONSE_HZ = (0, 25, 50, 75, 100, 150, 200, 300, 400, 600)


def load_weights(nt_mode, num_neurons, device):
    """Load the weights once. Only w_syn varies across a sweep."""
    if nt_mode == 'paper':
        weights = get_weights(
            str(path_con), str(path_comp), str(path_wt),
            csr=(device != 'mps'),
        ).to(device=device)
        return None, weights

    conn = pd.read_parquet(path_con)
    nt = load_nt_table(Path(path_wt) / 'neurotransmitters_783.csv', num_neurons)
    shipped = (
        conn.groupby('Presynaptic_Index')['Excitatory'].first()
        .reindex(range(num_neurons)).fillna(1).to_numpy()
    )
    force_tau = MODEL_PARAMS['tauSyn'] if nt_mode == 'signs' else None
    charge_ref = MODEL_PARAMS['tauSyn'] if nt_mode == 'channels-charge' else None
    taus, matrices, _ = build_channels(conn, nt, num_neurons, shipped,
                                       force_tau=force_tau,
                                       charge_ref_tau=charge_ref)
    matrices = [m.to(device=device) for m in matrices]

    if nt_mode == 'signs':
        return None, matrices[0]
    return taus, matrices


def build_model(nt_mode, taus, weights, batch, num_neurons, exc_indices,
                w_syn, device):
    """Instantiate the synapse model named by nt_mode with w_syn overridden."""
    params = dict(MODEL_PARAMS, wScale=w_syn)
    if nt_mode.startswith('channels'):
        return MultiNTTorchModel(
            batch, num_neurons, DT, params, taus, weights,
            LIFNeuron(batch, num_neurons, DT, params, device=device),
            exc_indices=exc_indices, device=device,
        )
    return TorchModel(batch, num_neurons, DT, params, weights,
                      exc_indices=exc_indices, device=device)


def measure_mn9(model, rates, mn9_index, num_steps, settle_steps, device):
    """Mean MN9 firing rate in Hz, averaged over the batch.

    Only MN9's own spikes are read back; transferring the full raster would
    cost more than the simulation and none of it is used here.
    """
    state = model.state_init()
    count = torch.zeros((), device=device)
    with torch.no_grad():
        for step in range(num_steps):
            state = model(rates, *state)
            if step >= settle_steps:
                count += state[2][:, mn9_index].sum()
    synchronize(device)

    measured_s = (num_steps - settle_steps) * DT / 1000.0
    return float(count.item()) / (rates.shape[0] * measured_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nt-mode', default='signs',
                        choices=('paper', 'signs', 'channels',
                                 'channels-charge'))
    parser.add_argument('--t-run', type=float, default=0.4)
    parser.add_argument('--n-run', type=int, default=16,
                        help='trials run in parallel; the Poisson input is '
                             'unseeded, so one trial is not a rate')
    parser.add_argument('--freq', type=float, default=CALIBRATION_HZ)
    parser.add_argument('--tol', type=float, default=0.05,
                        help='accept when within this fraction of the '
                             'reference rate')
    parser.add_argument('--max-iter', type=int, default=8)
    parser.add_argument('--reference-repeats', type=int, default=3,
                        help='repeats averaged into the reference rate; a '
                             'single measurement of it scatters by more than '
                             'the fitted shift it is being compared against')
    parser.add_argument('--dose-response', action='store_true',
                        help='sweep stimulus frequency instead of fitting')
    parser.add_argument('--noise', type=int, metavar='N',
                        help='repeat one measurement N times and report the '
                             'spread instead of fitting; a fitted w_syn means '
                             'nothing until compared against this')
    args = parser.parse_args()

    device = resolve_device()
    num_steps = int(args.t_run * 1000.0 / DT)
    settle_steps = int(SETTLE_MS / DT)
    if settle_steps >= num_steps:
        sys.exit('--t-run is shorter than the settling window')

    experiment = get_experiment()
    flyid2i, _ = get_hash_tables(str(path_comp))
    exc_indices = [flyid2i[n] for n in experiment['neu_exc']]
    mn9_index = flyid2i[MN9_LEFT]
    num_neurons = len(flyid2i)

    print(f'device        : {device}')
    print(f'synapse model : {args.nt_mode}')
    print(f'stimulus      : {len(exc_indices)} sugar GRNs at {args.freq} Hz')
    print(f'readout       : MN9 left ({MN9_LEFT})')
    print(f'trials        : {args.n_run} x {args.t_run}s, '
          f'first {SETTLE_MS:.0f}ms discarded')

    taus, weights = load_weights(args.nt_mode, num_neurons, device)

    def rate_at(w_syn, freq=args.freq):
        rates = torch.zeros(args.n_run, num_neurons, device=device)
        rates[:, exc_indices] = freq
        model = build_model(args.nt_mode, taus, weights, args.n_run,
                            num_neurons, exc_indices, w_syn, device)
        return measure_mn9(model, rates, mn9_index, num_steps,
                           settle_steps, device)

    if args.noise:
        # The Poisson input is unseeded, so repeats of an identical
        # configuration scatter. Any fitted shift smaller than this spread is
        # not a result.
        print(f'\nrepeating w_syn={MODEL_PARAMS["wScale"]:.4f} '
              f'{args.noise} times:')
        obs = []
        for i in range(args.noise):
            obs.append(rate_at(MODEL_PARAMS['wScale']))
            print(f'  [{i + 1:2d}] {obs[-1]:7.1f} Hz')
        mean = sum(obs) / len(obs)
        sd = (sum((x - mean) ** 2 for x in obs) / max(len(obs) - 1, 1)) ** 0.5
        print(f'\nmean {mean:.1f} Hz, sd {sd:.1f} Hz '
              f'({sd / mean * 100:.1f}% of mean), range '
              f'{min(obs):.1f}-{max(obs):.1f} Hz')
        return

    if args.dose_response:
        print(f'\n{"sugar Hz":>9} {"MN9 Hz":>9}')
        seen = {}
        for freq in DOSE_RESPONSE_HZ:
            seen[freq] = rate_at(MODEL_PARAMS['wScale'], freq)
            print(f'{freq:9d} {seen[freq]:9.1f}')
        peak = max(seen.values())
        if peak > 0 and args.freq in seen:
            print(f'\nMN9({args.freq:.0f} Hz) is {seen[args.freq] / peak * 100:.1f}% '
                  f'of the {peak:.1f} Hz peak over this sweep')
        return

    # Reference: the published model at the published w_syn. Measured under the
    # identical protocol so that the two numbers are comparable.
    print('\nreference (paper model, published w_syn):')
    ref_taus, ref_weights = load_weights('paper', num_neurons, device)
    ref_rates = torch.zeros(args.n_run, num_neurons, device=device)
    ref_rates[:, exc_indices] = args.freq
    ref_obs = []
    for _ in range(max(args.reference_repeats, 1)):
        ref_obs.append(measure_mn9(
            build_model('paper', ref_taus, ref_weights, args.n_run,
                        num_neurons, exc_indices, MODEL_PARAMS['wScale'],
                        device),
            ref_rates, mn9_index, num_steps, settle_steps, device,
        ))
    del ref_weights
    reference = sum(ref_obs) / len(ref_obs)
    spread = max(ref_obs) - min(ref_obs)
    print(f'  w_syn={MODEL_PARAMS["wScale"]:.4f} -> MN9 {reference:.1f} Hz '
          f'(mean of {len(ref_obs)}, spread {spread:.1f} Hz)')
    resolution = spread / reference if reference else 0.0

    if args.nt_mode == 'paper':
        print('\nnothing to fit: the reference is this model.')
        return

    # MN9's rate is not monotonic in w_syn over the whole range, so the bracket
    # is grown from the published value rather than assumed, and the search is
    # confined to the interval where the rate is still rising.
    lo, hi = MODEL_PARAMS['wScale'], MODEL_PARAMS['wScale']
    r_lo = r_hi = rate_at(lo)
    print(f'\nbracketing from the published w_syn ({r_lo:.1f} Hz):')
    for _ in range(6):
        if r_lo <= reference <= r_hi:
            break
        if r_hi < reference:
            hi *= 1.6
            r_hi = rate_at(hi)
            print(f'  w_syn={hi:.4f} -> {r_hi:7.1f} Hz')
        else:
            lo /= 1.6
            r_lo = rate_at(lo)
            print(f'  w_syn={lo:.4f} -> {r_lo:7.1f} Hz')
    else:
        print('\ncould not bracket the reference rate; MN9 may not be '
              'monotonic in w_syn over this range. Inspect with '
              '--dose-response and widen the search by hand.')
        return

    print('\nbisecting:')
    best = None
    for i in range(args.max_iter):
        mid = 0.5 * (lo + hi)
        r = rate_at(mid)
        err = abs(r - reference) / reference
        print(f'  [{i + 1:2d}] w_syn={mid:.4f} -> {r:7.1f} Hz  '
              f'({r / reference * 100:5.1f}% of reference)')
        if best is None or err < best[2]:
            best = (mid, r, err)
        if err <= args.tol:
            break
        if r < reference:
            lo = mid
        else:
            hi = mid

    w, r, err = best
    shift = abs(w - MODEL_PARAMS['wScale']) / MODEL_PARAMS['wScale']
    print(f'\nfitted w_syn = {w:.4f} mV   (MN9 {r:.1f} Hz vs reference '
          f'{reference:.1f} Hz, {err * 100:.1f}% off)')
    print(f'published    = {MODEL_PARAMS["wScale"]:.4f} mV   '
          f'ratio {w / MODEL_PARAMS["wScale"]:.3f}')
    if shift <= resolution:
        print(f'\nThe fitted shift ({shift * 100:.1f}%) is no larger than the '
              f'scatter of the reference itself ({resolution * 100:.1f}%). '
              f'This model needs no measurable change to w_syn; keep the '
              f'published value rather than reporting the fit as a result.')
    print('\nThe Poisson input is unseeded, so repeats scatter. Re-run to see '
          'the spread before trusting the last digit.')


if __name__ == '__main__':
    main()
