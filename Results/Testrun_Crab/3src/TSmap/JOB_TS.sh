#!/bin/bash
export EOS_MGM_URL=root://eos01.ihep.ac.cn/
procid=$1
segid=$[procid]
WorkDir=/home/lhaaso/hushicong/Standard_prog_lib/Source_Analysis/Space_energy_Joint_fitting/v0.99
exeprog=Src_TSMap
FitConfig=config/Testrun_Crab/Fit.yaml
Outdir=$WorkDir/Results/Testrun_Crab/3src/TSmap
[ -d $Outdir ] || mkdir -p $Outdir
$WorkDir/$exeprog $WorkDir/$FitConfig $segid $Outdir/TSmap_"$segid".root &> $Outdir/log_"$segid".txt
