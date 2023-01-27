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



function threepoint(tree, df, traits, N)
    #prefrom algortihm in Ho & Ane 2014
    #function estimaterates gets inputs into right form
    for i in 1:N
        #need to see if node is a tip (leaf)
        if isleaf(tree, df.Node[i])

            #where this leaf is in traits df and save trait
            data = filter(:species => x -> x == df.Node[i], traits)[1,2]
    
            df.logV[i] = log(df.t[i]) 
            df.p[i] = 1/df.t[i]
            df.yl[i] = data
            df.xl[i] = 1
            df.Q[i] = data/df.t[i]
            df.xx[i] = 1/df.t[i]
            df.yl[i] = data
            df.yy[i] =  (data*data)/df.t[i]
        else
            #need to find direct desendents 
            children = getchildren(tree, df.Node[i])
            cdf = filter(:Node => x -> x ∈ children, df)
    
            pA = sum(cdf.p)
            ws = cdf.p/pA
    
            df.logV[i] = sum(cdf.logV) + log(1+ pA) 
            df.p[i] = pA/(1+df.t[i]*pA)
            df.xl[i] = sum(ws.*cdf.xl)
            df.yl[i] = sum(ws.*cdf.yl)
            df.Q[i] = sum(cdf.Q) - ((df.t[i]*pA^2)/(1+df.t[i]*pA))*df.yl[i]*df.xl[i]
            df.xx[i] = sum(cdf.xx) - ((df.t[i]*pA^2)/(1+df.t[i]*pA))*df.xl[i]*df.xl[i]
            df.yy[i] = sum(cdf.yy) - ((df.t[i]*pA^2)/(1+df.t[i]*pA))*df.yl[i]*df.yl[i]
        end
    end
    return df
end

function estimaterates(tree, traits)
    #Returns evolution rate, starting value and negative log loglikelihood for traits on tip of tree
    #INPUTS
    #tree = tree with lengths, leaves all same length
    #traits = dataframe with leaf names and trait values, leafnames called 'species' and trait information called 'data'
    #OUTPUTS
    #sigmahat - evolution rate
    #betahat - estimated root trait value
    #negloglik - negative loglikelihood
    #df - dataframe of results - will add what all mean later

    #INPUT TESTS
    #traits for each leaf?

    names = getnodenames(tree, postorder)
    #total number of nodes
    N = length(names)
    #number of leaves
    n = nrow(traits)
    #vector of heigts for each node
    t = getheight.(tree, names)

    #dataframe to save information in
    df = DataFrame(Node = names, 
               t = t,
               logV = Vector{Union{Nothing, Float64}}(nothing, N),
               p = Vector{Union{Nothing, Float64}}(nothing, N),
               Q = Vector{Union{Nothing, Float64}}(nothing, N),
               xl = Vector{Union{Nothing, Float64}}(nothing, N),
               xx = Vector{Union{Nothing, Float64}}(nothing, N),
               yl = Vector{Union{Nothing, Float64}}(nothing, N),
               yy = Vector{Union{Nothing, Float64}}(nothing, N))

    df = threepoint(tree, df, traits, N)    

    betahat = inv(df.xx[N]) * df.Q[N]
    sigmahat = -(df.yy[N] - 2 * betahat * df.Q[N] + betahat * df.xx[N])/n

    
    if sigmahat < 0
        resdl = ones(n)*betahat - traits.data #replaces y which is traits
        traits2 = DataFrame(species = traits.species, data = resdl)
        df2 = threepoint(tree, df, traits2, N)
        sigmahat = df2.yy[N]/n
    end

    negloglik = (1/2)*(n*log(2*pi) + df.logV[N] + n*log(sigmahat))

    return betahat, sigmahat, negloglik, df
end  

#############################################################

#Test using Myrtaceae data
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

#only test with tmin for now
traits = dat[:,1:2]

#columns need renamed to work in my function
rename!(traits, :tmin => :data)

@btime estimaterates(tree1, traits)

#############################################################

#Compare to phylo networks results 

#load the tree using phylonetworks
tree2 = readTopology("Data/Qian2016.tree");

#get tip names
tipnames = tipLabels(tree2)

# Filter for species actually in the tree
missing_species = setdiff(tipnames, keep)
for i in eachindex(missing_species)
    deleteleaf!(tree2, missing_species[i])
end

#dataframe needs column called tipNames
dat.tipNames = dat.species;

@btime plm = phylolm(@formula(tmin ~ 1), dat, tree2)


#############################################################

#Bayes

#Create C 
leaves = dat.species
C = zeros(length(leaves), length(leaves));
#make the matrix (this code could probably be optimised)
for i in 1:length(leaves)
    C[i,i] = getheight(tree1, leaves[i])
    for j in i+1:length(leaves)
        ancestor = mrca(tree1, [leaves[i],leaves[j]])
        C[i,j] = getheight(tree1, ancestor)
        C[j,i] = C[i,j]
    end
end

#Create the model to go into sampler
@model function estmrates(C, trait)
    #C = length matrix, traits - trait we want to model
    z ~ Uniform(0,500)
    sigma ~ Uniform(0,100)
    trait ~ MvNormal(z*ones(length(trait)), sigma*C)
end

chain = sample(estmrates(C, dat.tmin), HMC(0.1, 5), 1000)

##########################################################

#PhyloNetworks lambda

plml = phylolm(@formula(tmin ~ 1), dat, tree2, model="lambda")

###########################################################

#Create the model to go into sampler
@model function estmrateslambda(C, trait)
    #C = length matrix, traits - trait we want to model
    z ~ Uniform(0,500)
    sigma ~ Uniform(0,100)
    lambda ~ Uniform(0,1)

    dC = diagm(diag(C))
    D = (lambda * (C - dC) + dC)

    trait ~ MvNormal(z*ones(length(trait)), sigma*(D))
end

chain = sample(estmrateslambda(C, dat.tmin), HMC(0.1, 5), 1000)