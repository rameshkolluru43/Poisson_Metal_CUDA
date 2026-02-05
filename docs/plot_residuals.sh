#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
out_dir="$repo_root/docs/residuals"
mkdir -p "$out_dir"

shopt -s nullglob
residual_files=( "$repo_root"/residual_history_*.csv )
if (( ${#residual_files[@]} == 0 )); then
  echo "No residual_history_*.csv files found in repo root: $repo_root" >&2
  exit 1
fi

if ! command -v gnuplot >/dev/null 2>&1; then
  echo "gnuplot not found. Please install it (e.g., 'brew install gnuplot')." >&2
  exit 1
fi

summary_file="$out_dir/summary.txt"
echo "Residuals summary (generated $(date))" > "$summary_file"
echo "" >> "$summary_file"

for f in "${residual_files[@]}"; do
  base="$(basename "${f%.*}")"
  rel_png="$out_dir/${base}_rel.png"
  l2_png="$out_dir/${base}_L2.png"

  gnuplot <<GNUPLOT
set datafile separator comma
set term pngcairo size 1400,900
set output '$rel_png'
set title '$base: relative residual'
set xlabel 'cycle'
set ylabel 'relative residual'
set grid
set logscale y
plot '$f' using 1:3 with linespoints lw 2 title 'rel_residual'
GNUPLOT

  gnuplot <<GNUPLOT
set datafile separator comma
set term pngcairo size 1400,900
set output '$l2_png'
set title '$base: L2 residual'
set xlabel 'cycle'
set ylabel 'L2 residual'
set grid
set logscale y
plot '$f' using 1:2 with linespoints lw 2 title 'residual_L2'
GNUPLOT

  # Extract first and last data rows (skip header)
  first_line=$(awk -F, 'NR==2 {print; exit}' "$f" || true)
  last_line=$(awk -F, 'NR>1 {line=$0} END{print line}' "$f" || true)
  if [[ -n "$first_line" && -n "$last_line" ]]; then
    IFS=',' read -r f_cycle f_l2 f_rel <<<"$first_line"
    IFS=',' read -r l_cycle l_l2 l_rel <<<"$last_line"
    # Compute ratios safely via awk to avoid bash float issues
    rel_ratio=$(awk -v a="$f_rel" -v b="$l_rel" 'BEGIN{ if (a>0) printf "%.4g", a/b; else print "n/a" }')
    l2_ratio=$(awk -v a="$f_l2" -v b="$l_l2" 'BEGIN{ if (a>0) printf "%.4g", a/b; else print "n/a" }')
    printf "%s\n  cycles: %s -> %s\n  rel_residual: %s -> %s (improvement x%s)\n  L2: %s -> %s (improvement x%s)\n\n" \
      "$base" "$f_cycle" "$l_cycle" "$f_rel" "$l_rel" "$rel_ratio" "$f_l2" "$l_l2" "$l2_ratio" \
      >> "$summary_file"
  else
    printf "%s\n  insufficient data rows to summarize\n\n" "$base" >> "$summary_file"
  fi
done

# Generate combined overlays by case pattern
plot_overlay() {
  local pattern="$1"   # e.g., residual_history_mg_default_*.csv
  local title="$2"     # e.g., MG default (rel)
  local ycol="$3"      # 3 for rel_residual, 2 for residual_L2
  local out_png="$4"   # output file path

  shopt -s nullglob
  local files=( "$repo_root"/$pattern )
  (( ${#files[@]} > 0 )) || return 0

  local script
  script="$(mktemp)"
  cat > "$script" <<GNUPLOT
set datafile separator comma
set term pngcairo size 1600,1000
set output '$out_png'
set title '$title'
set xlabel 'cycle'
set grid
set logscale y
set key outside right
GNUPLOT

  {
    printf "plot "
    local first=1
    for f in "${files[@]}"; do
      base=$(basename "${f%.*}")
      if (( first )); then
        first=0
      else
        printf ", "
      fi
      printf "'%s' using 1:%s with lines lw 2 title '%s'" "$f" "$ycol" "$base"
    done
    printf "\n"
  } >> "$script"

  gnuplot "$script"
  rm -f "$script"
}

# Overlays for relative residuals and L2 residuals
overlay_dir="$out_dir"
plot_overlay "residual_history_mg_default_*.csv" "MG default: relative residual overlays" 3 "$overlay_dir/combined_mg_default_rel.png"
plot_overlay "residual_history_mg_default_*.csv" "MG default: L2 residual overlays" 2 "$overlay_dir/combined_mg_default_L2.png"
plot_overlay "residual_history_mg_four_boundaries_*.csv" "MG four boundaries: relative residual overlays" 3 "$overlay_dir/combined_mg_four_boundaries_rel.png"
plot_overlay "residual_history_mg_four_boundaries_*.csv" "MG four boundaries: L2 residual overlays" 2 "$overlay_dir/combined_mg_four_boundaries_L2.png"

echo "Generated residual plots: $out_dir"
echo "Summary: $summary_file"
