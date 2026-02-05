
#!/bin/bash
set -e

# Change to the directory of this script so relative paths work regardless of invocation CWD
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Name of output binary
EXE=laplace_demo

echo "Compiling LaplaceMetalSolver with Metal support..."

clang++ -std=c++17 -fobjc-arc \
    metal/LaplaceMetalSolver_core.mm \
    metal/LaplaceMetalSolver_multigrid.mm \
    metal/LaplaceMetalSolver_solvers.mm \
    metal/LaplaceMetalSolver_io.mm \
    main_laplace_demo.cpp \
    -framework Metal -framework Foundation \
    -o $EXE

echo "Build finished. Running solver..."
# Auto-append manufactured fine-grid run unless user specified otherwise
ARGS=("$@")
has_manu=false
has_manu_n=false
for a in "$@"; do
    if [[ "$a" == "--manufactured" ]]; then has_manu=true; fi
    if [[ "$a" == --manufactured-n=* ]]; then has_manu_n=true; fi
done
if [ "$has_manu" = false ]; then
    ARGS+=("--manufactured")
fi
if [ "$has_manu_n" = false ]; then
    ARGS+=("--manufactured-n=513")
fi
./$EXE "${ARGS[@]}"
echo "Done. Solver finished."

# Prefer gnuplot from ../visualization/gnuplot, otherwise fallback to system gnuplot
GNUPLOT=""
if [ -x "../visualization/gnuplot" ]; then
    GNUPLOT="../visualization/gnuplot"
elif command -v gnuplot >/dev/null 2>&1; then
    GNUPLOT="gnuplot"
fi

if [ -n "$GNUPLOT" ]; then
    # Helper to plot a CSV matrix to a heatmap PNG with a consistent style
    plot_heatmap() {
        local csv="$1"
        local png="$2"
        local title="$3"
        cat > plot_cmds.gp <<EOF
set terminal pngcairo size 900,700 enhanced font 'Arial,10'
set output '$png'
set datafile separator ','
set pm3d map
unset key
set title '$title'
splot '$csv' matrix
EOF
        "$GNUPLOT" plot_cmds.gp && echo "Plotted heatmap $csv -> $png"
        rm -f plot_cmds.gp
    }

    # Helper to plot a CSV matrix to a 3D surface PNG
    plot_surface3d() {
        local csv="$1"
        local png="$2"
        local title="$3"
        cat > plot_cmds_3d.gp <<EOF
set terminal pngcairo size 1000,800 enhanced font 'Arial,10'
set output '$png'
set datafile separator ','
unset key
set title '$title (3D surface)'
set ticslevel 0
set view 60, 35
set hidden3d
set pm3d at s depthorder
set palette rgbformulae 33,13,10
splot '$csv' matrix with pm3d
EOF
        "$GNUPLOT" plot_cmds_3d.gp && echo "Plotted 3D surface $csv -> $png"
        rm -f plot_cmds_3d.gp
    }

    # Preserve original single default plot to solution.png (first available)
    OUTFILE=""
    if [ -f solution_mg.csv ]; then
        OUTFILE=solution_mg.csv
    elif [ -f solution_mg_four_boundaries.csv ]; then
        OUTFILE=solution_mg_four_boundaries.csv
    elif [ -f solution_four_boundaries.csv ]; then
        OUTFILE=solution_four_boundaries.csv
    elif [ -f solution.csv ]; then
        OUTFILE=solution.csv
    fi
    if [ -n "$OUTFILE" ]; then
    echo "Plotting $OUTFILE with $GNUPLOT..."
    plot_heatmap "$OUTFILE" "solution.png" "Laplace solution (heatmap)"
    plot_surface3d "$OUTFILE" "solution_3d.png" "Laplace solution"
    else
        echo "No solution CSV file found to plot. Expected one of: solution_mg.csv, solution_mg_four_boundaries.csv, solution_four_boundaries.csv, solution.csv."
    fi

    # Additionally, plot all recognized CSVs to dedicated files if they exist
    if [ -f solution_mg.csv ]; then
        plot_heatmap solution_mg.csv solution_mg.png "Multigrid (default)"
        plot_surface3d solution_mg.csv solution_mg_3d.png "Multigrid (default)"
    fi
    if [ -f solution_mg_four_boundaries.csv ]; then
        plot_heatmap solution_mg_four_boundaries.csv solution_mg_four_boundaries.png "Multigrid (four boundaries)"
        plot_surface3d solution_mg_four_boundaries.csv solution_mg_four_boundaries_3d.png "Multigrid (four boundaries)"
    fi
    if [ -f solution_four_boundaries.csv ]; then
        plot_heatmap solution_four_boundaries.csv solution_four_boundaries.png "Single-grid (four boundaries)"
        plot_surface3d solution_four_boundaries.csv solution_four_boundaries_3d.png "Single-grid (four boundaries)"
    fi
    if [ -f solution.csv ]; then
        plot_heatmap solution.csv solution_single.png "Single-grid (default)"
        plot_surface3d solution.csv solution_single_3d.png "Single-grid (default)"
    fi

    # Plot all benchmark/manufactured/other solution CSVs
    for csv in solution_*.csv mf_*.csv suite_*.csv; do
        if [ -f "$csv" ]; then
            base="${csv%.csv}"
            plot_heatmap "$csv" "$base.png" "$base"
            plot_surface3d "$csv" "${base}_3d.png" "$base"
        fi
    done
else
    echo "gnuplot not found; install or place the gnuplot binary in ../visualization/gnuplot to enable visualization."
fi