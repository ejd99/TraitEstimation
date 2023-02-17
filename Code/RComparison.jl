using RCall
using BenchmarkTools

R"
library('ape')
library('phylolm')
library('dplyr')

#read the tree
tre <- ape::read.tree(file = 'Data/Qian2016.tree')

#read the data
df <- read.csv(file = 'Data/Myrtaceae.csv');




#replace spaces with undersocres in dataframe species column
df$species<-gsub(' ', '_', df$species);




#Get mean for each species
dattib <- df %>% group_by(species) %>% summarize(tmin=mean(tmin), 
                                                tmax=mean(tmax), 
                                                trng=mean(trng), 
                                                stl1=mean(stl1), 
                                                stl2=mean(stl2), 
                                                stl3=mean(stl3), 
                                                stl4=mean(stl4), 
                                                swvl1=mean(swvl1), 
                                                swvl2=mean(swvl2), 
                                                swvl3=mean(swvl3), 
                                                swvl4=mean(swvl4), 
                                                ssr=mean(ssr), 
                                                tp=mean(tp))
dat <- as.data.frame(dattib)




#filter for only species in dat and tree
keep <- intersect(dat$species, tre$tip.label)



#remove unneeded
tree <- keep.tip(tre,keep)
dat <- dat[dat$species %in% keep,] 

#species need to be the row names
row.names(dat) <- dat$species
"

@btime R"
#fit model
fit = phylolm(tmin~1,data=dat,phy=tree)
"

@btime R"
#fit model
fit = phylolm(tmin~1,data=dat,phy=tree, model = c('lambda'))
"