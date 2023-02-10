using CSV
using DataFrames
using Phylo
using Optim
using LinearAlgebra
using Statistics
using Calculus
using BenchmarkTools
using PhyloNetworks
using GLM
using StatsBase
using JLD
using OnlineStats
using Turing
using StatsPlots
using Distributions


simpletree = parsenewick("(A:1,B:1,(C:0.5,D:0.5):0.5);")

plot(simpletree)

leafnames = ["D", "C", "A", "B"]
data = [1,2,3,4]

traits = DataFrame(species = leafnames, traits = data)

estimaterates(simpletree, traits)


tree2 = readTopology("(A:1,B:1,(C:0.5,D:0.5):0.5);")

dat = DataFrame(tipNames = leafnames, traits = data)

plm = phylolm(@formula(traits ~ 1), dat, tree2)