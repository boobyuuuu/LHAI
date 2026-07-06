#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import yaml

try:
    import ROOT
except ModuleNotFoundError as exc:
    raise SystemExit("PyROOT is required. Run this script after sourcing the LHAASO/ROOT environment.") from exc


PIXEL_SIZE_DEG = 0.1
NPIX = 64
CENTER_INDEX = NPIX // 2


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_workdir(fit_config, fit_path):
    workdir = Path(str(fit_config["WorkDir"]))
    if workdir.is_absolute():
        return workdir
    return (fit_path.parent / workdir).resolve()


def roi_center(fit_config):
    include = fit_config["ROI"]["Include"]
    shape = int(include[1])
    if shape == 0:
        return float(include[2]), float(include[3])
    if shape == 1:
        return (float(include[2]) + float(include[3])) / 2.0, (float(include[4]) + float(include[5])) / 2.0
    raise ValueError(f"Unsupported ROI shape flag: {shape}")


def template_edges(center_x, center_y):
    half_width = NPIX * PIXEL_SIZE_DEG / 2.0
    return center_x - half_width, center_x + half_width, center_y - half_width, center_y + half_width


def write_root(arr, root_path, center_x, center_y):
    x_min, x_max, y_min, y_max = template_edges(center_x, center_y)
    fout = ROOT.TFile(str(root_path), "RECREATE")
    if not fout or fout.IsZombie():
        raise RuntimeError(f"Cannot create ROOT file: {root_path}")

    for i in range(arr.shape[0]):
        hname = f"sample_{i}"
        h2 = ROOT.TH2D(hname, hname, NPIX, x_min, x_max, NPIX, y_min, y_max)
        img = arr[i, 0]
        for iy in range(NPIX):
            for ix in range(NPIX):
                h2.SetBinContent(ix + 1, iy + 1, float(img[iy, ix]))
        h2.Write()

    fout.Close()


def write_parinit(arr, parinit_path, template_file, dataset_name, epiv, f0_values, alpha):
    lines = [
        "DGE:",
        "  Active: 1",
        "  ConvoPSF: 1",
    ]

    for i in range(arr.shape[0]):
        f0 = f0_values[i]
        lines.extend([
            f"  Template{i}:",
            f"    Name: sample_{i}",
            f"    Tempfile: {template_file.as_posix()}",
            f"    TempHist: [sample_{i}]",
            f"    Epiv: {epiv:g}",
            "    SEDModel:",
            "      type: PL",
            f"      F0: [{f0:g}, 0, {max(abs(f0) * 10.0, 1.0):g}, 0, 1.e-15]",
            f"      alpha: [{alpha:g}, 1.0, 5.0, 0]",
        ])

    lines.extend([
        "",
        "SRC:",
        "  Active: 0",
        "",
        f"# Generated for {dataset_name} by Tools/NJU_AI/npy2root.py.",
        f"# Intrinsic templates are (N,1,{NPIX},{NPIX}) with {PIXEL_SIZE_DEG:g} deg pixels.",
        f"# The point-source pixel is numpy index (y={CENTER_INDEX}, x={CENTER_INDEX}).",
    ])

    parinit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert a 0.1 deg/pixel 64x64 NJU_AI NPY template set to ROOT and ParInit.yaml.")
    parser.add_argument("dataset_dir", nargs="?", default=".", help="Dataset directory containing Fit.yaml and samples.npy")
    parser.add_argument("--fit", default="Fit.yaml", help="Fit.yaml path relative to dataset_dir")
    parser.add_argument("--input", default="samples.npy", help="Input NPY path relative to dataset_dir")
    parser.add_argument("--root", default="samples.root", help="Output ROOT path relative to dataset_dir")
    parser.add_argument("--parinit", default="ParInit.yaml", help="Output ParInit.yaml path relative to dataset_dir")
    parser.add_argument("--epiv", type=float, default=50.0, help="Pivot energy written to each template component")
    parser.add_argument("--f0", type=float, default=1.0, help="Initial PL F0 value written to each template component")
    parser.add_argument("--alpha", type=float, default=3.0, help="Initial PL index written to each template component")
    parser.add_argument("--f0-list", default=None, help="Comma-separated per-template F0 values; overrides --f0")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    fit_path = dataset_dir / args.fit
    npy_path = dataset_dir / args.input
    root_path = dataset_dir / args.root
    parinit_path = dataset_dir / args.parinit

    fit_config = load_yaml(fit_path)
    workdir = resolve_workdir(fit_config, fit_path)
    center_x, center_y = roi_center(fit_config)

    arr = np.load(npy_path)
    if arr.ndim != 4 or arr.shape[1:] != (1, NPIX, NPIX):
        raise ValueError(f"Expected shape (N,1,{NPIX},{NPIX}), got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("Expected at least one template")

    write_root(arr, root_path, center_x, center_y)

    if args.f0_list is None:
        f0_values = [args.f0] * arr.shape[0]
    else:
        f0_values = [float(item) for item in args.f0_list.split(",") if item.strip()]
        if len(f0_values) != arr.shape[0]:
            raise ValueError(f"--f0-list has {len(f0_values)} values, but samples.npy has {arr.shape[0]} templates")

    outdir = Path(str(fit_config["Output"]["Outdir"]))
    template_file = workdir / outdir / root_path.name
    write_parinit(arr, parinit_path, template_file, dataset_dir.name, args.epiv, f0_values, args.alpha)

    print(f"Input NPY: {npy_path}")
    print(f"Input shape: {arr.shape}")
    print(f"ROI/template center from Fit.yaml: ({center_x}, {center_y})")
    print(f"Template edges: {template_edges(center_x, center_y)}")
    print(f"Wrote ROOT: {root_path}")
    print(f"Wrote ParInit: {parinit_path}")


if __name__ == "__main__":
    main()
