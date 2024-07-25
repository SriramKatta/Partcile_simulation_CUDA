#!/usr/bin/gnuplot

set terminal png
set output 'perf.png'
set title 'sim time vs numpoints'

set grid
set logscale y
set xlabel "Num particles"
set ylabel "mean sim time per timestep (millisec)"

plot "sim-datcutoff" with linespoints