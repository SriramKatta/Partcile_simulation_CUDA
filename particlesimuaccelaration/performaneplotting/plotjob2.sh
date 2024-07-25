#!/usr/bin/gnuplot

set terminal png
set output 'perf.png'
set title 'sim time vs numpoints'

set grid
set logscale y
set logscale x
set xlabel "Num particles"
set ylabel "mean sim time per timestep (millisec)"

set xtics rotate by 90 right

plot './newdat/sim-datcutoff' title "only cutoff" with linespoints, \
      './newdat/sim-datnbdaccv1' title "nbdacc" with linespoints,