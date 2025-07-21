# PracticalBlueprintThresholds: Code for thresholds in "Practical blueprint for low-depth photonic quantum computing with quantum dots"

This repository contains the code used to generate the noise thresholds and resource estimates presented in the paper "Practical blueprint for low-depth photonic quantum computing with quantum dots" by Ming Lai Chan, Aliki Anna Capatos, Peter Lodahl, Anders Søndberg Sørensen, and Stefano Paesani (preprint available on arXiv). The paper presents a practical blueprint for a low-optical-depth, emitter-based fault-tolerant photonic quantum computer, and includes realistic error modelling.

The code models the time-bin encoded generation of redundantly-encoded linear chain resource states that make up a larger fusion-based synchronous Foliated Floquet Colour Code (sFFCC) lattice. Errors are sampled at the resource state level, and mapped to the fusions of the lattice, capturing the effect of heralded losses and errors. The fusion attempts are repeated until success within $N$ trials. Thresholds are obtained with Monte-Carlo sampling.

The code has been developed using Linux (Ubuntu 22.04.5 LTS).
