#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import yaml

DATAGENERATING_DIR = Path(__file__).resolve().parent.parent
LOCAL_REPO_RESPONSE_DIR = Path(__file__).resolve().parents[3] / "response"
sys.path.insert(0, str(LOCAL_REPO_RESPONSE_DIR))
sys.path.insert(0, str(DATAGENERATING_DIR))

from generate_response import (  # noqa: E402
    DEFAULT_WORKDIR,
    FOV_DEG,
    NPIX,
    PIXEL_SIZE_DEG,
    make_fit,
    resolve_path,
    run,
    write_make_root,
    write_parinit,
    write_text,
)

SOURCE_TYPES = ("POINT", "GAUSSIAN", "DISK", "SHELL", "DIFFUSION")


def write_convert_6ch(path: Path) -> None:
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
print('Channels: 0=input, 1=excess, 2=bkg, 3=bkg_on, 4=poisson_on, 5=poissonexcess')
""",
    )


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def source_types_from_args(value: str) -> list[str]:
    if value.upper() == "ALL":
        return list(SOURCE_TYPES)
    selected = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(SOURCE_TYPES))
    if unknown:
        raise ValueError(f"Unknown source type(s): {unknown}")
    return selected


def run_response_for_type(args: argparse.Namespace, metadata: dict, source_type: str, exp_dir: Path) -> None:
    output_info = metadata["outputs"][source_type]
    samples = metadata["samples"][source_type]
    input_path = resolve_path(exp_dir, output_info["response_input_file"])
    output_path = resolve_path(exp_dir, output_info["expected_response_file"])
    work_root = output_path.with_suffix("").with_name(output_path.stem + "_work")
    work_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    src = np.load(input_path, mmap_mode="r")
    if src.ndim != 3 or src.shape[1:] != (NPIX, NPIX):
        raise ValueError(f"Expected source shape (N,64,64), got {src.shape}")
    if src.shape[0] != len(samples):
        raise ValueError(f"{source_type}: GT sample count {src.shape[0]} != metadata sample count {len(samples)}")
    rng = np.random.default_rng(args.seed + SOURCE_TYPES.index(source_type) * 1000003)
    roi_ra_centers = rng.uniform(0.0, 360.0, size=len(samples))
    roi_dec_centers = np.full(len(samples), args.dec_center, dtype=np.float64)

    response_files = []
    response_root = Path(args.workdir)
    response_metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "source_type": source_type,
        "num": len(samples),
        "batchsize": args.batchsize,
        "workdir": args.workdir,
        "energy": {
            "emin_log10_TeV": args.emin,
            "emax_log10_TeV": args.emax,
            "Epiv_TeV": args.epiv,
            "alpha": args.alpha,
        },
        "flux": {
            "source": "exp1_parameters.json response_F0 values",
            "order": "1.e-16",
            "coefficients_include_endpoints": True,
            "template_normalization": "GT arrays are unit-normalized; response flux is set only through F0",
        },
        "roi": {
            "shape": "rectangle",
            "fov_deg": FOV_DEG,
            "pixel_size_deg": PIXEL_SIZE_DEG,
            "ra_center_source": "random uniform [0,360) generated by generate_exp1_response.py seed",
            "dec_center_source": "fixed CLI --dec-center",
        },
        "samples": [],
        "batches": [],
        "channels": {
            "0": "input/intrinsic GT",
            "1": "excess = bkg_on - bkg",
            "2": "bkg",
            "3": "bkg_on",
            "4": "poisson_on",
            "5": "poissonexcess = poisson_on - bkg",
        },
    }

    for batch_id, start in enumerate(range(0, len(samples), args.batchsize)):
        stop = min(start + args.batchsize, len(samples))
        batch_samples_meta = samples[start:stop]
        batch_dir = work_root / f"batch_{start:05d}_{stop:05d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        np.save(batch_dir / "samples.npy", np.asarray(src[start:stop], dtype=np.float32)[:, None, :, :])

        if args.batchsize != 1:
            batch_ra = float(np.mean(roi_ra_centers[start:stop]))
            batch_dec = float(np.mean(roi_dec_centers[start:stop]))
        else:
            batch_ra = float(roi_ra_centers[start])
            batch_dec = float(roi_dec_centers[start])

        try:
            workdir_abs = Path(args.workdir).resolve()
            batch_abs = batch_dir.resolve()
            rel = batch_abs.relative_to(workdir_abs)
            outdir_rel = PurePosixPath(*rel.parts)
        except ValueError:
            outdir_rel = PurePosixPath("Tools", "NJU_AI", "DATAGENERATING", "Exp1_ParametricMorphologyReconstruction", work_root.name, batch_dir.name)

        fit = make_fit(args.workdir, outdir_rel, batch_ra, batch_dec, args.emin, args.emax)
        write_text(batch_dir / "Fit.yaml", yaml.safe_dump(fit, sort_keys=False))
        template_path = response_root / outdir_rel / "samples.root"
        flux_coeffs = [sample["response_F0"] for sample in batch_samples_meta]
        write_parinit(batch_dir / "ParInit.yaml", template_path, flux_coeffs, 16, args.epiv, args.alpha)
        write_parinit(batch_dir / "ParInit_background.yaml", template_path, np.zeros(stop - start), 16, args.epiv, args.alpha)
        write_make_root(batch_dir / "make_root.py")
        write_convert_6ch(batch_dir / "convert_root.py")

        for sample in batch_samples_meta:
            response_metadata["samples"].append(
                {
                    "index": sample["index"],
                    "batch": batch_id,
                    "flux_F0": sample["response_F0"],
                    "flux_order": "1.e-16",
                    "response_flux_unit": sample["response_flux_unit"],
                    "template_sum_64": sample["template_sum_64"],
                    "Epiv_TeV": args.epiv,
                    "alpha": args.alpha,
                    "roi_ra_center_deg": float(roi_ra_centers[sample["index"]]),
                    "roi_dec_center_deg": float(roi_dec_centers[sample["index"]]),
                    "Cx_deg": sample["Cx_deg"],
                    "Cy_deg": sample["Cy_deg"],
                    "scale_parameter": sample["scale_parameter"],
                    "scale_r39_deg": sample["scale_r39_deg"],
                    "morphology_parameters": sample["morphology_parameters"],
                }
            )
        response_metadata["batches"].append(
            {
                "batch": batch_id,
                "start": start,
                "stop": stop,
                "dir": str(batch_dir),
                "ra_center_deg": batch_ra,
                "dec_center_deg": batch_dec,
            }
        )

        if not args.dry_run:
            run([sys.executable, "make_root.py"], cwd=batch_dir)
            for label, out_name in [
                ("expectation", "expectation.root"),
                ("background", "background.root"),
                ("poisson", "poisson.root"),
            ]:
                fit_copy = dict(fit)
                fit_copy["Output"] = dict(fit["Output"])
                fit_copy["Output"]["fConExcess"] = out_name
                write_text(batch_dir / f"Fit_{label}.yaml", yaml.safe_dump(fit_copy, sort_keys=False))
            exe = Path(args.workdir) / "Tools" / "NJU_AI" / "Src_Convo_Template"
            run([str(exe), "Fit_expectation.yaml", "ParInit.yaml", "0", "0"], cwd=batch_dir)
            run([str(exe), "Fit_background.yaml", "ParInit_background.yaml", "0", "0"], cwd=batch_dir)
            run([str(exe), "Fit_poisson.yaml", "ParInit.yaml", "1", "0"], cwd=batch_dir)
            run([sys.executable, "convert_root.py", "--output", "response.npy"], cwd=batch_dir)
            response_files.append(batch_dir / "response.npy")

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(response_metadata, f, indent=2, ensure_ascii=False)

    if not args.dry_run:
        arrays = [np.load(path) for path in response_files]
        out = np.concatenate(arrays, axis=0)
        np.save(output_path, out)
        print(f"Wrote {output_path} shape={out.shape}")
        if not args.keep_work:
            for path in response_files:
                shutil.rmtree(path.parent)
    print(f"Wrote metadata {json_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate exact Exp1 LHAASO response arrays from exp1_parameters.json.")
    parser.add_argument("--types", default="ALL", help="ALL or comma-separated source types")
    parser.add_argument("--parameters", default="exp1_parameters.json")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--epiv", type=float, default=50.0)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--emin", type=float, default=1.4)
    parser.add_argument("--emax", type=float, default=3.4)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--dec-center", type=float, default=22.0)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    parameter_path = resolve_path(script_dir, args.parameters)
    exp_dir = parameter_path.parent
    metadata = load_metadata(parameter_path)
    for source_type in source_types_from_args(args.types):
        run_response_for_type(args, metadata, source_type, exp_dir)


if __name__ == "__main__":
    main()
