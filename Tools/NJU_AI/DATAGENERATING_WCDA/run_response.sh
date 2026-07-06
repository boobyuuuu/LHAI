#!/bin/bash
set -euo pipefail

ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        dataname|num|start_index|start-index|flux|fluxmin|fluxmax|fluxorder|fluxdist|output|batchsize|detector|wcda_nhit_min|wcda-nhit-min|wcda_nhit_max|wcda-nhit-max|Epiv|epiv|alpha|emin|emax|ra_center|ra-center|dec_center|dec-center|seed|workdir|outdir)
            KEY="$1"
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value for $KEY" >&2
                exit 2
            fi
            case "$KEY" in
                dataname) ARGS+=(--dataname "$1") ;;
                num) ARGS+=(--num "$1") ;;
                start_index|start-index) ARGS+=(--start-index "$1") ;;
                flux) ARGS+=(--flux "$1") ;;
                fluxmin) ARGS+=(--fluxmin "$1") ;;
                fluxmax) ARGS+=(--fluxmax "$1") ;;
                fluxorder) ARGS+=(--fluxorder "$1") ;;
                fluxdist) ARGS+=(--fluxdist "$1") ;;
                output) ARGS+=(--output "$1") ;;
                batchsize) ARGS+=(--batchsize "$1") ;;
                detector) ARGS+=(--detector "$1") ;;
                wcda_nhit_min|wcda-nhit-min) ARGS+=(--wcda-nhit-min "$1") ;;
                wcda_nhit_max|wcda-nhit-max) ARGS+=(--wcda-nhit-max "$1") ;;
                Epiv|epiv) ARGS+=(--epiv "$1") ;;
                alpha) ARGS+=(--alpha "$1") ;;
                emin) ARGS+=(--emin "$1") ;;
                emax) ARGS+=(--emax "$1") ;;
                ra_center|ra-center) ARGS+=(--ra-center "$1") ;;
                dec_center|dec-center) ARGS+=(--dec-center "$1") ;;
                seed) ARGS+=(--seed "$1") ;;
                workdir) ARGS+=(--workdir "$1") ;;
                outdir) ARGS+=(--outdir "$1") ;;
            esac
            shift
            ;;
        fluxshuffle|batch_time_seed|batch-time-seed|time_seed|time-seed|keep_work|dry_run)
            case "$1" in
                fluxshuffle) ARGS+=(--fluxshuffle) ;;
                batch_time_seed|batch-time-seed|time_seed|time-seed) ARGS+=(--batch-time-seed) ;;
                keep_work|keep-work) ARGS+=(--keep-work) ;;
                dry_run|dry-run) ARGS+=(--dry-run) ;;
            esac
            shift
            ;;
        --*)
            ARGS+=("$1")
            shift
            if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
                ARGS+=("$1")
                shift
            fi
            ;;
        *)
            echo "Unknown argument key: $1" >&2
            exit 2
            ;;
    esac
done

python3 generate_response.py "${ARGS[@]}"
