#!/usr/bin/env python3
import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import numpy as np
import yaml


DEFAULT_WORKDIR = "/home/lhaaso/zhliu/LZH/response"
NPIX = 64
PIXEL_SIZE_DEG = 0.1
FOV_DEG = NPIX * PIXEL_SIZE_DEG


def parse_optional_float(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "random", "false", "no"}:
        return None
    return float(value)


def resolve_path(base, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def make_time_seed(used_seeds):
    seed = time.time_ns() % 2147483647
    if seed <= 0:
        seed = 1
    while seed in used_seeds:
        seed += 1
        if seed >= 2147483647:
            seed = 1
    used_seeds.add(seed)
    return int(seed)


def sample_fluxes(num, fluxmin, fluxmax, dist, shuffle, seed, flux=None):
    if dist == 'const':
        if flux is None:
            raise ValueError('--flux must be provided when --fluxdist=const')
        return np.full(num, float(flux), dtype=np.float64)

    rng = np.random.default_rng(seed)
    if dist == 'uniform':
        fluxes = rng.uniform(fluxmin, fluxmax, size=num)
    elif dist == 'log_uniform':
        lo = math.log10(fluxmin)
        hi = math.log10(fluxmax)
        fluxes = 10 ** rng.uniform(lo, hi, size=num)
    else:
        raise ValueError(f'Unknown flux distribution: {dist}')

    if not shuffle:
        fluxes = np.sort(fluxes)

    return fluxes.astype(np.float64)


def sample_roi_centers(num, ra_center, dec_center, seed):
    rng = np.random.default_rng(seed)
    if ra_center is None:
        ra = rng.uniform(0.0, 360.0, size=num)
    else:
        ra = np.full(num, float(ra_center), dtype=np.float64)
    if dec_center is None:
        dec = rng.uniform(0.0, 90.0, size=num)
    else:
        dec = np.full(num, float(dec_center), dtype=np.float64)
    return ra, dec


def write_text(path, text):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def make_fit(workdir, outdir, ra_center, dec_center, emin, emax, detector, wcda_nhit_min, wcda_nhit_max):
    half = FOV_DEG / 2.0
    use_wcda = detector == 'wcda'
    use_km2a = detector == 'km2a'
    return {
        "WorkDir": str(workdir),
        "DataConfig": {
            "WCDA": "config/Data/WCDA/Cod/Data_20210305_20250731.yaml",
            "KM2A": "config/Data/KM2A/Data_20210720_20250731.yaml",
        },
        "CorOpt": 0,
        "DataUsed": {
            "SmoothBkg": 1,
            "WCDA": {
                "Active": 1 if use_wcda else 0,
                "NbinUsed": [int(wcda_nhit_min), int(wcda_nhit_max)],
                "ReBin": {"Active": 3, "Rebin": [2, 2, 1]},
            },
            "KM2A": {"Active": 1 if use_km2a else 0, "12_and_34": 0, "NbinUsed": [float(emin), float(emax)]},
        },
        "ROI": {
            "ROIfile": "none",
            "Include": [0, 1, round(ra_center - half, 6), round(ra_center + half, 6), round(dec_center - half, 6), round(dec_center + half, 6)],
            "Exclude": {"Active": 0, "Region": [0, 80.63, 22.02, 1, 1]},
        },
        "FastIteration": 0,
        "Fit": {"Fitting": 1, "FluxPoint": 0, "TS_Src": 0, "TS_Bin": 0, "FluxUL": 0},
        "TSmap": {"Active": 0, "WCDA": [0, 1, 6], "KM2A": [0, float(emin), float(emax)], "SrcID": 0, "Subtract": 0, "JOBScript": "JOB_TS.sh"},
        "Output": {"DrawOpt": 0, "Outdir": str(outdir), "fParResu": "none", "fConExcess": "expectation.root"},
    }


def write_parinit(path, template_path, fluxes, fluxorder, epiv, alpha):
    f0_order = f"1.e-{int(fluxorder)}"
    lines = ["DGE:", "  Active: 1", "  ConvoPSF: 1"]
    for i, f0 in enumerate(fluxes):
        upper = max(float(f0) * 10.0, 500.0)
        lines.extend([
            f"  Template{i}:",
            f"    Name: sample_{i}",
            f"    Tempfile: {template_path.as_posix()}",
            f"    TempHist: [sample_{i}]",
            f"    Epiv: {float(epiv):g}",
            "    SEDModel:",
            "      type: PL",
            f"      F0: [{float(f0):.8g}, 0, {upper:.8g}, 0, {f0_order}]",
            f"      alpha: [{float(alpha):.8g}, 1.0, 5.0, 0]",
        ])
    lines.extend(["", "SRC:", "  Active: 0", ""])
    write_text(path, "\n".join(lines))


def write_make_root(path):
    write_text(
        path,
        """#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np

try:
    import ROOT
except ModuleNotFoundError as exc:
    raise SystemExit('PyROOT is required. Source the LHAASO/ROOT environment first.') from exc

NPIX = 64
PIXEL = 0.1
parser = argparse.ArgumentParser()
parser.add_argument('--samples', default='samples.npy')
parser.add_argument('--fit', default='Fit.yaml')
parser.add_argument('--root', default='samples.root')
args = parser.parse_args()

import yaml
fit = yaml.safe_load(Path(args.fit).read_text(encoding='utf-8'))
inc = fit['ROI']['Include']
xc = 0.5 * (float(inc[2]) + float(inc[3]))
yc = 0.5 * (float(inc[4]) + float(inc[5]))
half = NPIX * PIXEL / 2.0
arr = np.load(args.samples)
if arr.ndim != 4 or arr.shape[1:] != (1, NPIX, NPIX):
    raise ValueError(f'Expected samples shape (N,1,64,64), got {arr.shape}')

fout = ROOT.TFile(str(args.root), 'RECREATE')
if not fout or fout.IsZombie():
    raise RuntimeError(f'Cannot create {args.root}')
for i in range(arr.shape[0]):
    h = ROOT.TH2D(f'sample_{i}', f'sample_{i}', NPIX, xc-half, xc+half, NPIX, yc-half, yc+half)
    img = arr[i, 0]
    for iy in range(NPIX):
        for ix in range(NPIX):
            h.SetBinContent(ix + 1, iy + 1, float(img[iy, ix]))
    h.Write()
fout.Close()
print(f'Wrote {args.root} with {arr.shape[0]} templates')
""",
    )


def write_convert(path):
    write_text(
        path,
        """#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np

try:
    import ROOT
except ModuleNotFoundError as exc:
    raise SystemExit('PyROOT is required. Source the LHAASO/ROOT environment first.') from exc

NPIX = 64
parser = argparse.ArgumentParser()
parser.add_argument('--samples', default='samples.npy')
parser.add_argument('--expectation', default='expectation.root')
parser.add_argument('--background', default='background.root')
parser.add_argument('--poisson', default='poisson.root')
parser.add_argument('--output', default='response.npy')
args = parser.parse_args()


def read_map(root_path, name):
    fin = ROOT.TFile.Open(str(root_path), 'READ')
    if not fin or fin.IsZombie():
        raise RuntimeError(f'Cannot open {root_path}')
    try:
        h = fin.Get(name)
        if not h:
            raise KeyError(f'Missing histogram {name} in {root_path}')
        img = np.zeros((NPIX, NPIX), dtype=np.float32)
        for iy in range(NPIX):
            for ix in range(NPIX):
                img[iy, ix] = float(h.GetBinContent(ix + 1, iy + 1))
        return img
    finally:
        fin.Close()


samples = np.load(args.samples).astype(np.float32)
out = np.zeros((samples.shape[0], 6, NPIX, NPIX), dtype=np.float32)
out[:, 0] = samples[:, 0]
for i in range(samples.shape[0]):
    hname = f'Non_exp_sample_{i}'
    bkg_on = read_map(args.expectation, hname)
    bkg = read_map(args.background, hname)
    poisson_on = read_map(args.poisson, hname)
    out[i, 1] = bkg_on - bkg
    out[i, 2] = bkg
    out[i, 3] = bkg_on
    out[i, 4] = poisson_on
    out[i, 5] = poisson_on - bkg
np.save(args.output, out)
print(f'Wrote {args.output} shape={out.shape}')
print('Channels: 0=intrinsic, 1=excess, 2=bkg, 3=bkg_on, 4=poisson, 5=on_off (poisson_on - bkg)')
""",
    )


def write_batch_scripts(batch_dir, source_dir, n_samples, batch_size):
    write_text(
        batch_dir / "run_one_chunk.sh",
        f"""#!/bin/bash
set -euo pipefail

START=${{1:?start index required}}
STOP=${{2:?stop index required}}
TAG=$(printf "%05d_%05d" "$START" "$STOP")

python3 make_samples.py --start "$START" --stop "$STOP"
python3 npy2root.py
python3 npy2root.py --f0 0 --parinit ParInit_background.yaml

python3 - <<'PYCONF'
from pathlib import Path
import yaml
fit = yaml.safe_load(Path('Fit.yaml').read_text(encoding='utf-8'))
fit['Output']['fConExcess'] = 'expectation.root'
with Path('Fit_expectation.yaml').open('w', encoding='utf-8', newline='\\n') as f:
    f.write(yaml.safe_dump(fit, sort_keys=False))
fit['Output']['fConExcess'] = 'background.root'
with Path('Fit_background.yaml').open('w', encoding='utf-8', newline='\\n') as f:
    f.write(yaml.safe_dump(fit, sort_keys=False))
fit['Output']['fConExcess'] = 'poisson.root'
with Path('Fit_poisson.yaml').open('w', encoding='utf-8', newline='\\n') as f:
    f.write(yaml.safe_dump(fit, sort_keys=False))
PYCONF

../../../Src_Convo_Template Fit_expectation.yaml ParInit.yaml 0 0
../../../Src_Convo_Template Fit_background.yaml ParInit_background.yaml 0 0
../../../Src_Convo_Template Fit_poisson.yaml ParInit.yaml 1 0
python3 root2npy.py

mkdir -p chunks
mv response.npy "chunks/response_${{TAG}}.npy"
mv response_intrinsic.npy "chunks/response_intrinsic_${{TAG}}.npy"
mv response_excess.npy "chunks/response_excess_${{TAG}}.npy"
mv response_bkg.npy "chunks/response_bkg_${{TAG}}.npy"
mv response_bkg_on.npy "chunks/response_bkg_on_${{TAG}}.npy"
rm -f samples.npy samples.root ParInit.yaml ParInit_background.yaml expectation.root background.root poisson.root Fit_expectation.yaml Fit_background.yaml Fit_poisson.yaml
""",
    )

    write_text(
        batch_dir / "run_all_chunks.sh",
        f"""#!/bin/bash
set -euo pipefail

N={n_samples}
CHUNK={batch_size}
for START in $(seq 0 $CHUNK $((N-1))); do
    STOP=$((START+CHUNK))
    if [ "$STOP" -gt "$N" ]; then STOP=$N; fi
    echo "Running {batch_dir.name} chunk $START:$STOP"
    bash run_one_chunk.sh "$START" "$STOP"
done

python3 merge_chunks.py
""",
    )

    write_text(
        batch_dir / "merge_chunks.py",
        """#!/usr/bin/env python3
from pathlib import Path
import numpy as np

chunk_dir = Path('chunks')
files = sorted(chunk_dir.glob('response_[0-9]*_[0-9]*.npy'))
if not files:
    raise SystemExit('No chunk response files found')
arrays = [np.load(p) for p in files]
out = np.concatenate(arrays, axis=0)
np.save('response_all.npy', out)
print(f'Wrote response_all.npy shape={out.shape}')
print('Channels: 0=intrinsic, 1=excess, 2=bkg, 3=bkg_on, 4=poisson')
""",
    )


def run(cmd, cwd, dry_run=False):
    print('+', ' '.join(str(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description='Generate LHAASO KM2A response datasets with catalog-like log-uniform fluxes.')
    p.add_argument('--dataname', required=True)
    p.add_argument('--num', type=int, required=True)
    p.add_argument('--start-index', type=int, default=0, help='Starting index in the input NPY file (default: 0)')
    p.add_argument('--fluxmin', type=float, default=0.1)
    p.add_argument('--fluxmax', type=float, default=10.0)
    p.add_argument('--flux', type=float, default=None, help='Single flux value, used when --fluxdist=const (all samples share this F0)')
    p.add_argument('--fluxorder', type=int, default=16)
    p.add_argument('--fluxdist', choices=['uniform', 'log_uniform', 'const'], default='log_uniform', help='Flux distribution: uniform, log_uniform, or const (all samples share --flux)')
    p.add_argument('--fluxshuffle', action='store_true', help='Shuffle flux values randomly; default is sorted ascending')
    p.add_argument('--output', required=True)
    p.add_argument('--batchsize', type=int, default=100)
    p.add_argument('--detector', choices=['km2a', 'wcda'], default='km2a')
    p.add_argument('--wcda-nhit-min', type=int, default=1)
    p.add_argument('--wcda-nhit-max', type=int, default=6)
    p.add_argument('--epiv', type=float, default=50.0)
    p.add_argument('--alpha', type=float, default=3.0)
    p.add_argument('--emin', type=float, default=1.4)
    p.add_argument('--emax', type=float, default=3.4)
    p.add_argument('--ra-center', default=None)
    p.add_argument('--dec-center', default='22')
    p.add_argument('--seed', type=int, default=20260514)
    p.add_argument('--batch-time-seed', action='store_true', help='Use the current time as the seed for each batch; flux/ROI sampling and Poisson use the same per-batch seed')
    p.add_argument('--workdir', default=DEFAULT_WORKDIR)
    p.add_argument('--outdir', default=None)
    p.add_argument('--keep-work', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if args.detector == 'wcda' and args.wcda_nhit_max < args.wcda_nhit_min:
        raise ValueError('--wcda-nhit-max must be greater than or equal to --wcda-nhit-min')

    base = Path(__file__).resolve().parent
    input_path = resolve_path(base, args.dataname)
    output_path = resolve_path(base, args.output)
    work_root = resolve_path(base, args.outdir) if args.outdir else output_path.with_suffix('').with_name(output_path.stem + '_work')
    work_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    src = np.load(input_path, mmap_mode='r')
    if src.ndim == 4 and src.shape[1:] == (1, NPIX, NPIX):
        src = src[:, 0, :, :]
    elif src.ndim == 3 and src.shape[1:] == (NPIX, NPIX):
        pass
    else:
        raise ValueError(f'Expected source shape (N,64,64) or (N,1,64,64), got {src.shape}')
    if args.start_index < 0 or args.start_index >= src.shape[0]:
        raise ValueError(f'--start-index {args.start_index} out of range [0, {src.shape[0]})')
    if args.start_index + args.num > src.shape[0]:
        raise ValueError(f'--start-index {args.start_index} + --num {args.num} = {args.start_index + args.num} exceeds source sample count {src.shape[0]}')

    ra_fixed = parse_optional_float(args.ra_center)
    dec_fixed = parse_optional_float(args.dec_center)
    if args.batch_time_seed:
        fluxes = np.empty(args.num, dtype=np.float64)
        ra_centers = np.empty(args.num, dtype=np.float64)
        dec_centers = np.empty(args.num, dtype=np.float64)
    else:
        fluxes = sample_fluxes(args.num, args.fluxmin, args.fluxmax, args.fluxdist, args.fluxshuffle, args.seed, flux=args.flux)
        ra_centers, dec_centers = sample_roi_centers(args.num, ra_fixed, dec_fixed, args.seed)
    used_time_seeds = set()

    metadata = {
        'input': str(input_path),
        'output': str(output_path),
        'start_index': args.start_index,
        'num': args.num,
        'batchsize': args.batchsize,
        'detector': args.detector,
        'wcda_nhit_used': [args.wcda_nhit_min, args.wcda_nhit_max] if args.detector == 'wcda' else None,
        'km2a_energy_used_log10_TeV': [args.emin, args.emax] if args.detector == 'km2a' else None,
        'workdir': args.workdir,
        'seed': {'mode': 'batch_time' if args.batch_time_seed else 'fixed', 'base': args.seed},
        'energy': {'emin_log10_TeV': args.emin, 'emax_log10_TeV': args.emax, 'Epiv_TeV': args.epiv, 'alpha': args.alpha},
        'flux': {'distribution': args.fluxdist, 'min': args.fluxmin, 'max': args.fluxmax, 'value': args.flux, 'order': f'1.e-{args.fluxorder}', 'shuffle': args.fluxshuffle, 'seed': 'batch_time' if args.batch_time_seed else args.seed},
        'roi': {'shape': 'rectangle', 'fov_deg': FOV_DEG, 'pixel_size_deg': PIXEL_SIZE_DEG, 'ra_center_mode': 'random' if ra_fixed is None else 'fixed', 'dec_center_mode': 'random' if dec_fixed is None else 'fixed'},
        'samples': [],
        'batches': [],
        'channels': {'0': 'intrinsic/GT', '1': 'excess', '2': 'bkg', '3': 'bkg_on', '4': 'poisson', '5': 'on_off (poisson_on - bkg)'},
    }

    response_files = []
    response_root = base.parents[2]
    for batch_id, start in enumerate(range(0, args.num, args.batchsize)):
        stop = min(start + args.batchsize, args.num)
        batch_dir = work_root / f'batch_{start:05d}_{stop:05d}'
        batch_dir.mkdir(parents=True, exist_ok=True)

        if args.batch_time_seed:
            batch_seed = make_time_seed(used_time_seeds)
            fluxes[start:stop] = sample_fluxes(stop - start, args.fluxmin, args.fluxmax, args.fluxdist, args.fluxshuffle, batch_seed, flux=args.flux)
            batch_ra, batch_dec = sample_roi_centers(stop - start, ra_fixed, dec_fixed, batch_seed)
            ra_centers[start:stop] = batch_ra
            dec_centers[start:stop] = batch_dec
        else:
            batch_seed = args.seed if args.seed == 0 else args.seed + batch_id

        batch_samples = np.asarray(src[args.start_index + start:args.start_index + stop], dtype=np.float32)[:, None, :, :]
        np.save(batch_dir / 'samples.npy', batch_samples)

        batch_ra = float(np.mean(ra_centers[start:stop]))
        batch_dec = float(np.mean(dec_centers[start:stop]))
        try:
            rel = batch_dir.resolve().relative_to(response_root.resolve())
            outdir_rel = PurePosixPath(*rel.parts)
        except ValueError:
            outdir_rel = PurePosixPath('Tools', 'NJU_AI', 'DATAGENERATING', work_root.name, batch_dir.name)

        fit = make_fit(
            args.workdir,
            outdir_rel,
            batch_ra,
            batch_dec,
            args.emin,
            args.emax,
            args.detector,
            args.wcda_nhit_min,
            args.wcda_nhit_max,
        )
        write_text(batch_dir / 'Fit.yaml', yaml.safe_dump(fit, sort_keys=False))
        template_path = Path(args.workdir) / outdir_rel / 'samples.root'
        write_parinit(batch_dir / 'ParInit.yaml', template_path, fluxes[start:stop], args.fluxorder, args.epiv, args.alpha)
        write_parinit(batch_dir / 'ParInit_background.yaml', template_path, np.zeros(stop - start), args.fluxorder, args.epiv, args.alpha)
        write_make_root(batch_dir / 'make_root.py')
        write_convert(batch_dir / 'convert_root.py')
        write_batch_scripts(batch_dir, input_path.parent, args.num, args.batchsize)

        for sample_index in range(start, stop):
            metadata['samples'].append({
                'index': args.start_index + sample_index,
                'batch': batch_id,
                'seed': int(batch_seed),
                'flux_F0': float(fluxes[sample_index]),
                'flux_order': f'1.e-{args.fluxorder}',
                'Epiv_TeV': float(args.epiv),
                'alpha': float(args.alpha),
                'ra_center_deg': float(ra_centers[sample_index]),
                'dec_center_deg': float(dec_centers[sample_index]),
            })

        metadata['batches'].append({'batch': batch_id, 'start': start, 'stop': stop, 'dir': str(batch_dir), 'seed': int(batch_seed), 'poisson_seed': int(batch_seed), 'ra_center_deg': batch_ra, 'dec_center_deg': batch_dec})

        if not args.dry_run:
            run([sys.executable, 'make_root.py'], cwd=batch_dir)
            for source_fit, out_name in [('expectation', 'expectation.root'), ('background', 'background.root'), ('poisson', 'poisson.root')]:
                fit_copy = dict(fit)
                fit_copy['Output'] = dict(fit['Output'])
                fit_copy['Output']['fConExcess'] = out_name
                write_text(batch_dir / f'Fit_{source_fit}.yaml', yaml.safe_dump(fit_copy, sort_keys=False))
            exe = Path(args.workdir) / 'Tools' / 'NJU_AI' / 'Src_Convo_Template'
            run([str(exe), 'Fit_expectation.yaml', 'ParInit.yaml', '0', str(batch_seed)], cwd=batch_dir)
            run([str(exe), 'Fit_background.yaml', 'ParInit_background.yaml', '0', str(batch_seed)], cwd=batch_dir)
            run([str(exe), 'Fit_poisson.yaml', 'ParInit.yaml', '1', str(batch_seed)], cwd=batch_dir)
            run([sys.executable, 'convert_root.py', '--output', 'response.npy'], cwd=batch_dir)
            response_files.append(batch_dir / 'response.npy')

    json_path = output_path.with_suffix('.json')
    with json_path.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if not args.dry_run:
        arrays = [np.load(p) for p in response_files]
        out = np.concatenate(arrays, axis=0)
        np.save(output_path, out)
        print(f'Wrote {output_path} shape={out.shape}')
        if not args.keep_work:
            for p in response_files:
                shutil.rmtree(p.parent)
    print(f'Wrote metadata {json_path}')


if __name__ == '__main__':
    main()
