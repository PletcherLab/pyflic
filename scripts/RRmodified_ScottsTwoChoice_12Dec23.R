rm(list=ls())
#source("FLICUserFunctions.R")
#source("MiscFunctions.R")
library(ggplot2)
library(stats)
library(gridExtra)
library(reshape2)
library(dplyr)
library(tidyr)
library(gtools)

#attach("FLICFUNCTIONS",2)
attach("FLICFunctions", pos=2)

p<-ParametersClass.TwoWell()
p<-SetParameter(p,Feeding.Event.Link.Gap=5)
p<-SetParameter(p,Feeding.Threshold=10)
p<-SetParameter(p,Feeding.Minimum=10)
p<-SetParameter(p,Tasting.Interval=c(0,10))
#p<-SetParameter(p,Correct.For.Dual.Feeding = FALSE)
p.sucrose.left<-p

p<-ParametersClass.TwoWell()
p<-SetParameter(p,Feeding.Event.Link.Gap=5)
p<-SetParameter(p,Feeding.Threshold=10)
p<-SetParameter(p,Feeding.Minimum=10)
p<-SetParameter(p,Tasting.Interval=c(0,10))
p<-SetParameter(p,PI.Multiplier=-1)
#p<-SetParameter(p,Correct.For.Dual.Feeding = FALSE)
p.sucrose.right<-p

monitors<-c(2,3,4,5,6,7,9,10)
ed<-read.csv("ExpDesign.csv") 

## Because of the PI multiplier, all data for the 'A' wells are sucrose and those for 'B' are yeast.

p.list<-list(p.sucrose.left,p.sucrose.left,p.sucrose.right,p.sucrose.right,p.sucrose.left,p.sucrose.left,p.sucrose.right,p.sucrose.right)

fs6<-Feeding.Summary.Monitors(monitors,p.list,ed,range=c(0,360),SaveToFile=TRUE,TransformLicks=TRUE,filename="FeedingSummary_6hrs_Tf_SP")
fs18<-Feeding.Summary.Monitors(monitors,p.list,ed,range=c(360,1440),SaveToFile=TRUE,TransformLicks=TRUE,filename="FeedingSummary_L18hrs_Tf_SP")
fs<-Feeding.Summary.Monitors(monitors,p.list,ed,SaveToFile=TRUE,TransformLicks=TRUE,filename="FeedingSummary_24hrs_Tf_SP")

## Raw plots for determining DFM baseline drift

RawDataPlot.DFM(DFM2,OutputPNGFile=TRUE)
RawDataPlot.DFM(DFM3,OutputPNGFile=TRUE)
RawDataPlot.DFM(DFM4,OutputPNGFile=TRUE)
RawDataPlot.DFM(DFM5,OutputPNGFile=TRUE)
RawDataPlot.DFM(DFM6,OutputPNGFile=TRUE)
RawDataPlot.DFM(DFM7,OutputPNGFile=TRUE)
RawDataPlot.DFM(DFM9,OutputPNGFile=TRUE)
RawDataPlot.DFM(DFM10,OutputPNGFile=TRUE)

save.image("LoadedREnvironFile")

## Binned Feeding Summaries

ed2<-read.csv("ExpDesign2.csv")
bfs<-BinnedFeeding.Summary.Monitors(monitors,p.list,ed2,binsize.min=30,SaveToFile=TRUE,TransformLicks=FALSE,filename="BinnedSummary_noTf_24hrs")
bfs.events<-BinnedFeeding.Summary.Monitors.ByEvents(monitors,p.list,ed2,binsize.Enum=10,SaveToFile=TRUE,TransformLicks=FALSE,filename="BinnedSummaryByEvents_noTf_24hrs")

############### And the Graphing begins!!!

library(ggplot2)
fs6<-read.csv("COMBINED_Rmv_FeedingSummary_6hrs_Tf_SP_Data.csv")
fs<-read.csv("COMBINED_Rmv_FeedingSummary_24hrs_Tf_SP_Data.csv")

## Separate treatment into two the light/dark condition (Treatment) and S2/S20 (Sucrose)

require(tidyr)
fs6<-separate(data = fs6, col = Treatment, sep="_", into = c("Treatment","Sucrose"))
fs<-separate(data = fs, col = Treatment, sep="_", into = c("Treatment","Sucrose"))

##Remove flies from dataframe that don't eat
index<-!(fs6$LicksA < 0.00001 | fs6$LicksB < 0.00001)
fs6<-fs6[index,]

index<-!(fs6$MedDurationA > 13)
fs6<-fs6[index,]

index<-!(fs6$EventsB > 150)
fs6<-fs6[index,]

##############################################
## Hedonic Feeding Graph - plotting MedDurationA will show sucrose durations

## Unweighted Sucrose MedDuration: this is the usual plotting mechanism
ggplot(fs,aes(Sucrose,MedDurationA,fill=Sucrose)) + 
  geom_jitter (aes(x = Sucrose, y = MedDurationA, color=Sucrose),width=0.25,size=3) +
  geom_jitter (aes(x = Treatment, y = MedDurationA, color=Treatment, shape=Experiment),width=0.25,size=3) +
  stat_summary(fun=mean, color="#333333", shape=13, size=0.5) +
  stat_summary(fun.data=mean_se, geom="errorbar", size=0.35) +
  scale_color_manual(values=c("#292836","#52A08D","#C1CEDA","#494856")) + 
  facet_wrap("Treatment") +
  scale_color_brewer(palette="RdGy") +
  theme_bw(base_size=20) +
  scale_x_discrete(limits=c("fixed","S2","S20","S2_arab")) + 
  xlab("Sucrose") +
  ylab("Sucrose MedDuration") +
  ylim(0,8) +
  ggtitle("24hrs GtACR > PAM03 Sucrose MedDuration 11Dec23")
   + annotate("text",x=1,y=3,label="Anova, p=0.233")
ggsave("Rplot_24hrs_68311GtACR_SucroseMedDuration.png", width = 10, height = 6)

## Weighted Sucrose MedDuration: With the size proportional to the weights/number of events.
ggplot(fs,aes(Sucrose,MedDurationA,fill=Sucrose)) + 
  geom_jitter (aes(x = Sucrose, y = MedDurationA, color=Sucrose, size=EventsA, shape=Experiment),width=0.25) +
  #geom_jitter (aes(x = Treatment, y = MedDurationA, color=Treatment, shape=Experiment, size=EventsA),width=0.25) +
  stat_summary(fun=mean, color="#333333", shape=13) +
  stat_summary(fun.data=mean_se, geom="errorbar", size=0.35) +
  scale_color_manual(values=c("#292836","#52A08D","#C1CEDA","#494856")) + 
  facet_wrap("Treatment") +
  #scale_color_brewer(palette="RdGy") +
  theme_bw(base_size=20) +
  #scale_x_discrete(limits=c("fixed","S2","S20","S2_arab")) + 
  xlab("Sucrose") +
  ylab("Median Event Duration") +
  #ylim(0,8) +
  ggtitle("COMBINED 24hrs GtACR > MB a'b' a Sucrose MedDuration 29n31Jan25")
ggsave("Rplot_COMBINED_24hrs_68370dGtACR_SucroseMedDuration.png", width = 10, height = 6)

############ STATS - Weighted and Unweighted
## Unweighted Two-way ANOVA (Sucrose MedDurations)
anova.unweighted<-(aov(MedDurationA~Experiment+Sucrose+Treatment+Treatment:Sucrose,data=fs))
summary(anova.unweighted)

## Weighted Two-way ANOVA
anova.weighted<-(aov(MedDurationA~Experiment+Sucrose+Treatment+Treatment:Sucrose,weights=EventsA,data=fs))
summary(anova.weighted)

##############################################
## Homeostatic Feeding Graphs

##Single replicate!

s2.data<-subset(fs6,fs6$Sucrose=="S2")
tmp<-s2.data[,c("Treatment","EventsB")]
Food<-rep("Yeast",nrow(tmp))
tmp<-data.frame(tmp,Food)
names(tmp)<-c("Treatment","Events","Food")

s2.data<-s2.data[,c("Treatment","EventsA")]
Food<-rep("S2",nrow(tmp))
s2.data<-data.frame(s2.data,Food)
names(s2.data)<-c("Treatment","Events","Food")

s2.data<-rbind(s2.data,tmp)

##Multiple replicates!

s2.data<-subset(fs,fs$Sucrose=="S2")
tmp<-s2.data[,c("Treatment","EventsB","Experiment")]
Food<-rep("Yeast",nrow(tmp))
tmp<-data.frame(tmp,Food)
names(tmp)<-c("Treatment","Events","Experiment","Food")

s2.data<-s2.data[,c("Treatment","EventsA","Experiment")]
Food<-rep("S2",nrow(tmp))
s2.data<-data.frame(s2.data,Food)
names(s2.data)<-c("Treatment","Events","Experiment","Food")

s2.data<-rbind(s2.data,tmp)

## Unweighted S2 vs Yeast Event Plot - Homeostatic Hunger
ggplot(s2.data,aes(Food,Events,fill=Food)) + 
  geom_jitter (aes(x = Food, y = Events, color=Food,shape=Experiment),width=0.25,size=3) +
  stat_summary(fun=mean, color="#333333", shape=13, size=0.5) +
  stat_summary(fun.data=mean_se, geom="errorbar", size=0.35) +
  scale_color_manual(values=c("#292836","#52A08D","#C1CEDA","#494856")) + 
  facet_wrap("Treatment") +
  #scale_color_brewer(palette="RdGy") +
  theme_bw(base_size=20) +
  #scale_x_discrete(limits=c("fixed","S2","S20","S2_arab")) + 
  xlab("Food") +
  ylab("Events") +
  #ylim(0,8) +
  ggtitle("COMBINED 24hrs GtACR > MB a'b' a S2 v Yeast Events 29n31Jan25")
#+ annotate("text",x=1,y=3,label="Anova, p=0.233")
ggsave("Rplot_COMBINED_24hrs_68370GtACR_S2vYeastEvents.png", width = 10, height = 6)

############ STATS - Weighted and Unweighted
## Unweighted Two-way ANOVA (S2 v Yeast Events)
anova.unweighted2<-(aov(Events~Experiment+Food+Treatment+Treatment:Food,data=s2.data))
summary(anova.unweighted2)

##############################################
## Linear Reg: change in MedDuration for BOTH light and dark as a function of how long the light was on for Graphs

ggplot(fs) + 
  geom_jitter (aes(x = EventsA, y = MedDurationA, color=Treatment), size=3, width=0.25) +
  #scale_color_manual(values="#292836") +
  geom_smooth(aes(x=EventsA,y=MedDurationA, color=Treatment), method='lm') +
  #geom_jitter (aes(x = Treatment, y = MedDurationA, color=Treatment, shape=Experiment, size=EventsA),width=0.25) +
  #stat_summary(fun=mean, color="#333333", shape=13) +
  #stat_summary(fun.data=mean_se, geom="errorbar", size=0.35) +
  scale_color_manual(values=c("#8F9779","#292836","#23bd01","#0B6623")) + 
  #facet_wrap("Treatment") +
  #scale_color_brewer(palette="RdGy") +
  theme_bw(base_size=20) +
  xlab("Sucrose Events") +
  ylab("Sucrose Median Duration") +
  #ylim(0,8) +
  ggtitle("COMBINED 24hrs GtACR > PAM06 4n6Sept24")
ggsave("Rplot_COMBINED_24hrs_68302dGtACR_EventvDurReg.png", width = 10, height = 6)

##############################################
## Sucrose Events Graph - plotting EventsA will show sucrose events

## Unweighted Sucrose Events
ggplot(fs,aes(Sucrose,EventsA,fill=Sucrose)) + 
  geom_jitter (aes(x = Sucrose, y = EventsA, color=Sucrose,shape=Experiment),width=0.25,size=3) +
  #geom_jitter (aes(x = Treatment, y = MedDurationA, color=Treatment, shape=Experiment),width=0.25,size=3) +
  stat_summary(fun=mean, color="#333333", shape=13, size=0.5) +
  stat_summary(fun.data=mean_se, geom="errorbar", size=0.35) +
  scale_color_manual(values=c("#292836","#52A08D","#C1CEDA","#494856")) + 
  facet_wrap("Treatment") +
  #scale_color_brewer(palette="RdGy") +
  theme_bw(base_size=20) +
  #scale_x_discrete(limits=c("fixed","S2","S20","S2_arab")) + 
  xlab("Sucrose") +
  ylab("Sucrose Events") +
  #ylim(0,8) +
  ggtitle("24hrs GtACR > PAM06 Sucrose Events 4n6Sept24")
# + annotate("text",x=1,y=3,label="Anova, p=0.233")
ggsave("Rplot_COMBINED_24hrs_68302GtACR_SucroseEvents.png", width = 10, height = 6)

##############################################
## MedDuration for all foods

##Multiple replicates!
#collect S2 vs yeast data
s2.data<-subset(fs,fs$Sucrose=="S2")
tmp<-s2.data[,c("Treatment","MedDurationB","Experiment")]
Food<-rep("Y.S2",nrow(tmp))
tmp<-data.frame(tmp,Food)
names(tmp)<-c("Treatment","MedDuration","Experiment","Food")

s2.data<-s2.data[,c("Treatment","MedDurationA","Experiment")]
Food<-rep("S2",nrow(tmp))
s2.data<-data.frame(s2.data,Food)
names(s2.data)<-c("Treatment","MedDuration","Experiment","Food")

s2.data<-rbind(s2.data,tmp)

#collect S20 vs yeast data
s20.data<-subset(fs,fs$Sucrose=="S20")
tmp2<-s20.data[,c("Treatment","MedDurationB","Experiment")]
Food<-rep("Y.S20",nrow(tmp2))
tmp2<-data.frame(tmp2,Food)
names(tmp2)<-c("Treatment","MedDuration","Experiment","Food")

s20.data<-s20.data[,c("Treatment","MedDurationA","Experiment")]
Food<-rep("S20",nrow(tmp2))
s20.data<-data.frame(s20.data,Food)
names(s20.data)<-c("Treatment","MedDuration","Experiment","Food")

s20.data<-rbind(s20.data,tmp2)

#combine S2 and S20 data
all.data<-rbind(s2.data,s20.data)

## Unweighted S2, S20, Yeast MedDuration Plot
ggplot(all.data,aes(Food,MedDuration,fill=Food)) + 
  geom_jitter (aes(x = Food, y = MedDuration, color=Food,shape=Experiment),width=0.25,size=3) +
  stat_summary(fun=mean, color="#333333", shape=13, size=0.5) +
  stat_summary(fun.data=mean_se, geom="errorbar", size=0.35) +
  scale_color_manual(values=c("#292836","#52A08D","#C1CEDA","#494856")) + 
  facet_wrap("Treatment") +
  #scale_color_brewer(palette="RdGy") +
  theme_bw(base_size=20) +
  #scale_x_discrete(limits=c("fixed","S2","S20","S2_arab")) + 
  xlab("Food") +
  ylab("MedDuration") +
  #ylim(0,8) +
  ggtitle("COMBINED 24hrs GtACR > MBON26 Food Duration 20n22May24")
#+ annotate("text",x=1,y=3,label="Anova, p=0.233")
ggsave("Rplot_COMBINED_24hrs_87907GtACR_FoodDuration.png", width = 10, height = 6)


##############################################
##Plotting binned data by time #FIX TO BE COMPATABLE WITH DATA NAMES

bfs<-read.csv("BinnedSummary_noTf_24hrs_Stats.csv")

## Separate treatment column
require(tidyr)
bfs<-separate(data = bfs, col = Treatment, sep="_", into = c("Treatment","Food"))

## Sucrose Lick comparison
data_long <- bfs %>%
  gather(LickType, Licks, LicksA:LicksB)

data_long$LickType[data_long$LickType=="LicksA"]<-bfs$Food
data_long$LickType[data_long$LickType=="LicksB"]<- "Yeast"

## Sucrose Event comparison
data_long <- bfs %>%
  gather(EventType, Events, EventsA:EventsB)


data_long$EventType[data_long$EventType=="EventsA"]<-bfs$Food
data_long$EventType[data_long$EventType=="EventsB"]<-"Yeast"

## Sucrose MedDuration comparison
data_long <- bfs %>%
  gather(DurType, MedDuration, MedDurationA,MedDurationB)


data_long$DurType[data_long$DurType=="MedDurationA"]<-bfs$Food
data_long$DurType[data_long$DurType=="MedDurationB"]<-"Yeast"


##Plotting binned data
pd <- position_dodge(5) 
ggplot(data_long,aes(x=Minutes,y=MedDuration,color=DurType,group=DurType)) + 
  #geom_errorbar(aes(ymin=Events-EventsSEM, ymax=Events+EventsSEM,color=Treatment), width=.1, position=pd) +
  geom_line(position=pd,size=1) +
  geom_point(position=pd, size=1.5, shape=21, aes(fill=DurType)) +
  scale_color_manual(values=c("deepskyblue4","paleturquoise3","grey25","deeppink4")) +
  scale_fill_manual(values=c("deepskyblue4","paleturquoise3","grey25","deeppink4")) +
  theme_bw(base_size=20) +
  facet_wrap("Treatment") +
  xlab("Time (in min)") +
  ylab("MedDuration") +
  ggtitle("Binned MedDuration 11Dec23") +
  #annotate("text",x=1,y=3,label="Anova, p=0.233")
  ggsave("BinnedMedDuration_11Dec23_24hrs.png", width = 10, height = 6)

##Plotting binned data PIs
pd <- position_dodge(5) 
ggplot(bfs,aes(x=Minutes,y=EventPI,color=Food,group=Food)) + 
  #geom_errorbar(aes(ymin=Events-EventsSEM, ymax=Events+EventsSEM,color=Treatment), width=.1, position=pd) +
  geom_line(position=pd,size=1) +
  geom_point(position=pd, size=1.5, shape=21, aes(fill=Food)) +
  scale_color_manual(values=c("deepskyblue4","paleturquoise3","grey25","deeppink4")) +
  scale_fill_manual(values=c("deepskyblue4","paleturquoise3","grey25","deeppink4")) +
  theme_bw(base_size=20) +
  facet_wrap("Treatment") +
  xlab("Time (in min)") +
  ylab("Event PI") +
  ggtitle("Binned Event PI 11Dec23") +
  #annotate("text",x=1,y=3,label="Anova, p=0.233")
  ggsave("BinnedEventPI_11Dec23_24hrs.png", width = 10, height = 6)


####################################
## Additional Graphs - for fun :) ##
####################################

## S20 vs Yeast Events Graphing - for additional information

## SINGLE Replicate Experiment

s20.data<-subset(fs,fs$Sucrose=="S20")
tmp<-s20.data[,c("Treatment","EventsB")]
Food<-rep("Yeast",nrow(tmp))
tmp<-data.frame(tmp,Food)
names(tmp)<-c("Treatment","Events","Food")

s20.data<-s20.data[,c("Treatment","EventsA")]
Food<-rep("Sucrose",nrow(tmp))
s20.data<-data.frame(s20.data,Food)
names(s20.data)<-c("Treatment","Events","Food")

s20.data<-rbind(s20.data,tmp)

ggplot(s20.data,aes(Food,Events,fill=Food)) + 
  geom_jitter (aes(x = Food, y = Events, color=Food),width=0.25,size=3) +
  stat_summary(fun=mean, color="#333333", shape=13, size=0.5) +
  stat_summary(fun.data=mean_se, geom="errorbar", size=0.35) +
  scale_color_manual(values=c("#292836","#52A08D","#C1CEDA","#494856")) + 
  facet_wrap("Treatment") +
  #scale_color_brewer(palette="RdGy") +
  theme_bw(base_size=20) +
  #scale_x_discrete(limits=c("fixed","S2","S20","S2_arab")) + 
  xlab("Food") +
  ylab("Events") +
  #ylim(0,8) +
  ggtitle("S2 Treatment Food Events")
#+ annotate("text",x=1,y=3,label="Anova, p=0.233")
ggsave("Rplot_Trial_S20Events.png", width = 10, height = 6)


##Plot ALL Yeast Events for flies - separated by food and treatment

ggplot(fs,aes(Sucrose,EventsB,fill=Sucrose)) + 
  geom_jitter (aes(x = Sucrose, y = EventsB, color=Sucrose),width=0.25,size=3) +
  stat_summary(fun=mean, color="#333333", shape=13, size=0.5) +
  stat_summary(fun.data=mean_se, geom="errorbar", size=0.35) +
  scale_color_manual(values=c("#292836","#52A08D","#C1CEDA","#494856")) + 
  facet_wrap("Treatment") +
  #scale_color_brewer(palette="RdGy") +
  theme_bw(base_size=20) +
  #scale_x_discrete(limits=c("fixed","S2","S20","S2_arab")) + 
  xlab("Treatment") +
  ylab("Yeast Events") +
  #ylim(0,8) +
  ggtitle("Trial Yeast Events 16June22")
#+ annotate("text",x=1,y=3,label="Anova, p=0.233")
ggsave("Rplot_Trial_YeastEvents.png", width = 10, height = 6)

## If you also want to plot yeast licks, then use LicksB in the code above
