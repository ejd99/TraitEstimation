using Distributed

addprocs(3)

#load packages
@everywhere begin
    using Pkg; Pkg.activate(".") #makes sure all workers use this directory 
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
end

#load a faff with data
@everywhere begin
    #load the data
    df = DataFrame(CSV.File("Data/Myrtaceae.csv"))

    #load the tree using Phylo
    tree1 = open(parsenewick, "Data/Qian2016.tree")

    #have the full tree, need to filter for the data we have.
    #remove missing species from dataframe
    dropmissing!(df, :species)

    #add underscores to dataframe
    df.species = replace.(df.species, " " => "_")

    #want species in tree that we have data for
    keep = intersect(getleafnames(tree1), df.species)
    keeptips!(tree1, keep)

    #filter dataframe
    filter!(:species => x -> x ∈ keep, df)

    #use mean value for each species
    gdf = groupby(df, :species)
    dat = combine(gdf, [:tmin, :tmax, :trng, :stl1, :stl2, :stl3, :stl4, :swvl1, :swvl2, :swvl3, :swvl4, :ssr, :tp] .=> mean; renamecols=false)

    
end

#Create C 
@everywhere begin
    #Create C 
    leavess = dat.species
    C = zeros(length(leavess), length(leavess));
    #make the matrix (this code could probably be optimised)
    for i in 1:length(leavess)
        C[i,i] = getheight(tree1, leavess[i])
        for j in i+1:length(leavess)
            ancestor = mrca(tree1, [leavess[i],leavess[j]])
            C[i,j] = getheight(tree1, ancestor)
            C[j,i] = C[i,j]
        end
    end
end

#Create model for tmin
@everywhere begin
    @model function estmratestmin(C, trait)
        #C = length matrix, traits - trait we want to model
        z ~ Uniform(0,500)
        sigma ~ Uniform(0,100)
        trait ~ MvNormal(z*ones(length(trait)), sigma*C)
    end
end

sample(estmratestmin(C, dat.tmin), HMC(0.01, 10), MCMCDistributed(), 1000)