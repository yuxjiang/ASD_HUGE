# ASD_Hum_Genet.
Code for the ASD project submitted to Human Genetics

## What's in the folder

This repo contains:
* C++ implementation of the functional-flow algorithm.
* Matlab scripts that generates figures in the manuscript.

## External dependencies

The C++ implementation relies on [armadillo](http://arma.sourceforge.net).
To compile the code, please modify the Makefile(s) by changing the path to which armadillo is installed if necessary.

## libcb-CPP

This is the dependency library that the implementation of functional-flow relied on. Please build the program by running `make` under `libcb-CPP/build` (remember to change the path configuration in Makefile if necessary).

## cpp

The implemenation of functional-flow and some other binary program that are used in this manuscript.
In particular, after building the binaries, you should be able to see the following program under `cpp/bin`.

* `make-net.run` - the one used to build a network file from plain text file of pcc. The pcc file is a three column file having
  ```
  <gene_id_A> <gene_id_B> <correlation>
  ```
* `merge-net.run` - the one used to merge Gene expression network with PPI network.
* `nbp.run` - the functional-flow implementation
* `tail-stat.run` - the one computes the statistic on the high risk tails (combines gene score and vairant scores)

## matlab

This folder contains Matlab scripts that we used to generate figures in the manuscript.
