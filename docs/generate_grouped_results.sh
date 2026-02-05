#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
repo_root=".."
out_tex="grouped_results.tex"

# Escape LaTeX caption text
escape_caption() {
  printf '%s' "$1" | sed \
    -e 's/\\/\\\\textbackslash{}/g' \
    -e 's/_/\\\\_/g' \
    -e 's/%/\\\\%/g' \
    -e 's/&/\\\\&/g' \
    -e 's/#/\\\\#/g' \
    -e 's/\$/\\\\\$/g' \
    -e 's/{/\\\\{/g' \
    -e 's/}/\\\\}/g'
}

begin_section() {
  local title="$1"
  echo "\\subsection{$(escape_caption "$title")}" >> "$out_tex"
}

emit_triptych() {
  local sol_img="$1"; shift
  local rel_img="$1"; shift
  local l2_img="$1"; shift
  local caption="$1"

  echo "\\begin{figure}[ht]" >> "$out_tex"
  echo "  \\centering" >> "$out_tex"
  if [[ -n "$sol_img" ]]; then
    echo "  \\begin{subfigure}[b]{0.32\\textwidth}" >> "$out_tex"
    echo "    \\centering" >> "$out_tex"
    echo "    \\IfFileExists{$sol_img}{\\includegraphics[width=\\linewidth]{$sol_img}}{\\fbox{$(escape_caption "$sol_img") not found}}" >> "$out_tex"
    echo "    \\caption{Solution}" >> "$out_tex"
    echo "  \\end{subfigure}" >> "$out_tex"
  fi
  if [[ -n "$rel_img" ]]; then
    echo "  \\begin{subfigure}[b]{0.32\\textwidth}" >> "$out_tex"
    echo "    \\centering" >> "$out_tex"
    echo "    \\IfFileExists{$rel_img}{\\includegraphics[width=\\linewidth]{$rel_img}}{\\fbox{$(escape_caption "$rel_img") not found}}" >> "$out_tex"
    echo "    \\caption{Relative residual}" >> "$out_tex"
    echo "  \\end{subfigure}" >> "$out_tex"
  fi
  if [[ -n "$l2_img" ]]; then
    echo "  \\begin{subfigure}[b]{0.32\\textwidth}" >> "$out_tex"
    echo "    \\centering" >> "$out_tex"
    echo "    \\IfFileExists{$l2_img}{\\includegraphics[width=\\linewidth]{$l2_img}}{\\fbox{$(escape_caption "$l2_img") not found}}" >> "$out_tex"
    echo "    \\caption{L2 residual}" >> "$out_tex"
    echo "  \\end{subfigure}" >> "$out_tex"
  fi
  echo "  \\caption{$(escape_caption "$caption")}" >> "$out_tex"
  echo "\\end{figure}" >> "$out_tex"
}

echo "% Auto-generated grouped results (solution+residuals)" > "$out_tex"

# Multigrid categories (avoid associative arrays for macOS bash compatibility)
category_title() {
  case "$1" in
    mg_default) echo "Multigrid: default BCs" ;;
    mg_four_boundaries) echo "Multigrid: four boundaries" ;;
    *) echo "$1" ;;
  esac
}

shopt -s nullglob
categories=( "mg_default" "mg_four_boundaries" )
for category in "${categories[@]}"; do
  # Collect sizes from residual CSVs
  csvs=( "$repo_root"/residual_history_${category}_*.csv )
  (( ${#csvs[@]} > 0 )) || continue
  begin_section "$(category_title "$category")"
  for csv in "${csvs[@]}"; do
    size_with_ext="${csv##residual_history_${category}_}"
    size="${size_with_ext%.csv}"
    # Solution image candidates in repo root
    sol_candidates=(
      "$repo_root/solution_${category}_${size}.png"
      "$repo_root/solution_${category}_${size}_ds.png"
      "$repo_root/solution_${category}_${size}_3d.png"
    )
    sol_img=""
    for s in "${sol_candidates[@]}"; do
      [[ -f "$s" ]] && { sol_img="$(basename "$s")"; break; }
    done
    # Residual plot images in docs/residuals
    rel_img="residuals/residual_history_${category}_${size}_rel.png"
    l2_img="residuals/residual_history_${category}_${size}_L2.png"
    # Only emit if any exists
    if [[ -n "$sol_img" || -f "$rel_img" || -f "$l2_img" ]]; then
      emit_triptych "$sol_img" "$rel_img" "$l2_img" "${category} ${size}"
      echo "" >> "$out_tex"
    fi
  done
done

# Manufactured: group solutions (no residual histories available)
shopt -s nullglob
mf_candidates=( "$repo_root"/mf_*.png )
if (( ${#mf_candidates[@]} > 0 )); then
  begin_section "Manufactured solutions"
  # Build unique base names without _3d suffix
  bases=$(for img in "${mf_candidates[@]}"; do basename "${img%.png}" | sed 's/_3d$//'; done | sort -u)
  for base in $bases; do
    sol1="$repo_root/${base}.png"
    sol3d="$repo_root/${base}_3d.png"
    sol_img=""
    [[ -f "$sol1" ]] && sol_img="$(basename "$sol1")"
    echo "\\begin{figure}[ht]" >> "$out_tex"
    echo "  \\centering" >> "$out_tex"
    if [[ -n "$sol_img" ]]; then
      echo "  \\begin{subfigure}[b]{0.48\\textwidth}" >> "$out_tex"
      echo "    \\centering" >> "$out_tex"
      echo "    \\IfFileExists{$sol_img}{\\includegraphics[width=\\linewidth]{$sol_img}}{\\fbox{$(escape_caption "$sol_img") not found}}" >> "$out_tex"
      echo "    \\caption{Solution}" >> "$out_tex"
      echo "  \\end{subfigure}" >> "$out_tex"
    fi
    if [[ -f "$sol3d" ]]; then
      b3="$(basename "$sol3d")"
      echo "  \\begin{subfigure}[b]{0.48\\textwidth}" >> "$out_tex"
      echo "    \\centering" >> "$out_tex"
      echo "    \\IfFileExists{$b3}{\\includegraphics[width=\\linewidth]{$b3}}{\\fbox{$(escape_caption "$b3") not found}}" >> "$out_tex"
      echo "    \\caption{3D surface}" >> "$out_tex"
      echo "  \\end{subfigure}" >> "$out_tex"
    fi
    echo "  \\caption{$(escape_caption "$base")}" >> "$out_tex"
    echo "\\end{figure}" >> "$out_tex"
    echo "" >> "$out_tex"
  done
fi

echo "Generated $out_tex"
