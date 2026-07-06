#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

try:
    import ROOT
except ModuleNotFoundError as exc:
    raise SystemExit("PyROOT is required. Run this script after sourcing the LHAASO/ROOT environment.") from exc


NPIX = 64


def read_histogram(fin, name):
    h2 = fin.Get(name)
    if not h2:
        raise KeyError(f"ROOT histogram not found: {name}")
    nx = h2.GetNbinsX()
    ny = h2.GetNbinsY()
    if nx != NPIX or ny != NPIX:
        raise ValueError(f"{name} has shape {ny}x{nx}; expected {NPIX}x{NPIX}")

    img = np.zeros((NPIX, NPIX), dtype=np.float32)
    for iy in range(NPIX):
        for ix in range(NPIX):
            img[iy, ix] = float(h2.GetBinContent(ix + 1, iy + 1))
    return img


def read_root_map(root_path, hist_name):
    fin = ROOT.TFile.Open(str(root_path), "READ")
    if not fin or fin.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {root_path}")
    try:
        return read_histogram(fin, hist_name)
    finally:
        fin.Close()


def hist_name(index, mode):
    if mode in {"sum", "energy-bins"}:
        return f"Non_exp_sample_{index}"
    if mode == "bin-diagonal":
        return f"Non_exp_{index}_sample_{index}"
    raise ValueError(f"Unsupported response mode: {mode}")


def save_channel_files(output_path, out):
    stem = output_path.with_suffix("")
    names = ["intrinsic", "excess", "bkg", "bkg_on"]
    for idx, name in enumerate(names):
        np.save(stem.with_name(f"{stem.name}_{name}.npy"), out[:, idx : idx + 1])


def main():
    parser = argparse.ArgumentParser(description="Convert Src_Convo_Template ROOT output to explicit intrinsic/excess/bkg/bkg-on NPY maps.")
    parser.add_argument("dataset_dir", nargs="?", default=".", help="Dataset directory containing samples.npy and response ROOT files")
    parser.add_argument("--input", default="samples.npy", help="Original intrinsic NPY path relative to dataset_dir")
    parser.add_argument("--root", default="convo_response.root", help="Src_Convo_Template ROOT output path relative to dataset_dir")
    parser.add_argument("--background-root", default=None, help="Optional background-only ROOT path relative to dataset_dir")
    parser.add_argument("--poisson-root", default=None, help="Optional Poisson-fluctuated ROOT path relative to dataset_dir")
    parser.add_argument("--output", default="convo_response.npy", help="Output NPY path relative to dataset_dir")
    parser.add_argument(
        "--response-mode",
        choices=["sum", "bin-diagonal", "energy-bins"],
        default="sum",
        help="sum reads one ROOT file; energy-bins reads response_bin_i.root files; bin-diagonal reads Non_exp_i_sample_i",
    )
    parser.add_argument("--save-channel-files", action="store_true", help="Also save *_intrinsic.npy, *_excess.npy, *_bkg.npy, and *_bkg_on.npy")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    input_path = dataset_dir / args.input
    root_path = dataset_dir / args.root
    output_path = dataset_dir / args.output

    arr = np.load(input_path)
    if arr.ndim != 4 or arr.shape[1:] != (1, NPIX, NPIX):
        raise ValueError(f"Expected intrinsic shape (N,1,{NPIX},{NPIX}), got {arr.shape}")

    n_channels = 5 if args.poisson_root else 4
    out = np.zeros((arr.shape[0], n_channels, NPIX, NPIX), dtype=np.float32)
    out[:, 0] = arr[:, 0].astype(np.float32)

    if args.response_mode == "energy-bins":
        for i in range(arr.shape[0]):
            bkg_on = read_root_map(dataset_dir / f"response_bin_{i}.root", hist_name(i, args.response_mode))
            background_root_path = dataset_dir / f"background_bin_{i}.root"
            if background_root_path.exists():
                bkg = read_root_map(background_root_path, "Non_exp_sample_0")
            else:
                bkg = np.zeros((NPIX, NPIX), dtype=np.float32)
            out[i, 1] = bkg_on - bkg
            out[i, 2] = bkg
            out[i, 3] = bkg_on
    else:
        background_root_path = dataset_dir / args.background_root if args.background_root else None
        poisson_root_path = dataset_dir / args.poisson_root if args.poisson_root else None
        for i in range(arr.shape[0]):
            bkg_on = read_root_map(root_path, hist_name(i, args.response_mode))
            if background_root_path is not None:
                bkg = read_root_map(background_root_path, hist_name(i, args.response_mode))
            elif arr.shape[0] > 1:
                bkg = read_root_map(root_path, hist_name(1, args.response_mode))
            else:
                bkg = np.zeros((NPIX, NPIX), dtype=np.float32)
            out[i, 1] = bkg_on - bkg
            out[i, 2] = bkg
            out[i, 3] = bkg_on
            if poisson_root_path is not None:
                out[i, 4] = read_root_map(poisson_root_path, hist_name(i, args.response_mode))

    np.save(output_path, out)
    if args.save_channel_files:
        save_channel_files(output_path, out)

    print(f"Input NPY: {input_path}")
    print(f"ROOT input: {root_path if args.response_mode != 'energy-bins' else dataset_dir / 'response_bin_<i>.root'}")
    print(f"Response mode: {args.response_mode}")
    print(f"Wrote NPY: {output_path}")
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")
    if args.poisson_root:
        print("Channels: 0=intrinsic, 1=excess, 2=bkg, 3=bkg_on, 4=poisson")
    else:
        print("Channels: 0=intrinsic, 1=excess, 2=bkg, 3=bkg_on")


if __name__ == "__main__":
    main()
