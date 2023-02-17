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
            df.yy[i] =  (data*data)/df.t[i]
        else
            #need to find direct desendents 
            children = getchildren(tree, df.Node[i])
            cdf = filter(:Node => x -> x ∈ children, df)
    
            pA = sum(cdf.p)
            ws = cdf.p/pA
    
            df.logV[i] = sum(cdf.logV) + log(1+ df.t[i]*pA) 
            df.p[i] = pA/(1+df.t[i]*pA)
            df.xl[i] = sum(ws.*cdf.xl)
            df.yl[i] = sum(ws.*cdf.yl)
            df.Q[i] = sum(cdf.Q) - ((df.t[i]*pA^2)/(1+df.t[i]*pA))*df.xl[i]*df.yl[i]
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

    leafnames = getnodenames(tree, postorder)
    #total number of nodes
    N = length(leafnames)
    #number of leaves
    n = nrow(traits)

    #dataframe to save information in
    df = DataFrame(Node = leafnames, 
            t = Vector{Union{Nothing, Float64}}(nothing, N),
            logV = Vector{Union{Nothing, Float64}}(nothing, N),
            p = Vector{Union{Nothing, Float64}}(nothing, N),
            Q = Vector{Union{Nothing, Float64}}(nothing, N),
            xl = Vector{Union{Nothing, Float64}}(nothing, N),
            xx = Vector{Union{Nothing, Float64}}(nothing, N),
            yl = Vector{Union{Nothing, Float64}}(nothing, N),
            yy = Vector{Union{Nothing, Float64}}(nothing, N))

    for i in 1:(N-1)
        par = getparent(tree, df.Node[i])
        df.t[i] = distance(tree, df.Node[i], par)
    end

    df.t[N] = 0

    df = threepoint(tree, df, traits, N)    

    betahat = inv(df.xx[N]) * df.Q[N]
    sigmahat = (df.yy[N] - 2 * betahat * df.Q[N] + betahat * df.xx[N] * betahat)/n

    #=
    if sigmahat < 0 #if used prints df2 at the end?
        resdl = ones(n)*betahat - traits.data #replaces y which is traits
        traits2 = DataFrame(species = traits.species, data = resdl)
        df2 = threepoint(tree, df, traits2, N)
        sigmahat = df2.yy[N]/n
    end
    =#
    
    negloglik = (1/2)*(n*log(2*pi) + df.logV[N] + n + n*log(sigmahat))

    return betahat, sigmahat, negloglik, df
end  

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


###############################################################
#3 Point Signal

function estimaterates(tree, traits, lambda::Float64)
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

    leafnames = getnodenames(tree, postorder)
    #total number of nodes
    N = length(leafnames)
    #number of leaves
    n = nrow(traits)

    #dataframe to save information in
    df = DataFrame(Node = leafnames, 
            t = Vector{Union{Nothing, Float64}}(nothing, N),
            logV = Vector{Union{Nothing, Float64}}(nothing, N),
            p = Vector{Union{Nothing, Float64}}(nothing, N),
            Q = Vector{Union{Nothing, Float64}}(nothing, N),
            xl = Vector{Union{Nothing, Float64}}(nothing, N),
            xx = Vector{Union{Nothing, Float64}}(nothing, N),
            yl = Vector{Union{Nothing, Float64}}(nothing, N),
            yy = Vector{Union{Nothing, Float64}}(nothing, N))


    #get the heights and multiply internal nodes heighst by lambda
    heights = DataFrame(Node = leafnames,
                        t = Vector{Union{Nothing, Float64}}(nothing, N))

    for i in 1:N
        if isleaf(tree, leafnames[i])
            heights.t[i] = getheight(tree, leafnames[i])
        else
            heights.t[i] = lambda*getheight(tree, leafnames[i])
        end
    
    end

    for i in 1:(N-1)
        #t is the distance between the node and its parent
        par = getparent(tree, df.Node[i])
        df.t[i] = heights.t[i] - first(heights[heights.Node .== par, 2])
    end


    df.t[N] = 0

    df = threepoint(tree, df, traits, N)    

    betahat = inv(df.xx[N]) * df.Q[N]
    sigmahat = (df.yy[N] - 2 * betahat * df.Q[N] + betahat * df.xx[N] * betahat)/n

    #=
    if sigmahat < 0 #if used prints df2 at the end?
        resdl = ones(n)*betahat - traits.data #replaces y which is traits
        traits2 = DataFrame(species = traits.species, data = resdl)
        df2 = threepoint(tree, df, traits2, N)
        sigmahat = df2.yy[N]/n
    end
    =#
    
    negloglik = (1/2)*(n*log(2*pi) + df.logV[N] + n + n*log(sigmahat))

    logV = df.logV[N]

    return betahat, sigmahat, negloglik, logV, df
end  

@btime estimaterates(tree1, traits, 0.8825229)

tooptimise(tree1, traits, [0.8])

function tooptimise(tree, traits, lambda)
    res = estimaterates(tree, traits, lambda[1])
    n = nrow(traits)

    ll = (1/2)*(n*log(2*pi) + res[4] + n*log(res[2]))

    return ll
end

lower = [0.5]
upper = [1.0]
start = [0.8]

opts = optimize(x -> tooptimise(tree1, traits, x), lower, upper, start)