#!/bin/bash

dirRemote=/home/lhaaso/hushicong/Standard_prog_lib/Source_Analysis/Space_energy_Joint_fitting/Developing/v0.99_beta
dirLocal=`pwd`

cp $dirRemote/Makefile* $dirLocal
date=`date`
echo " Update from $dirRemote" >> $dirLocal/Update.log
echo " Update : $date" >> $dirLocal/Update.log
if [ ! -f $dirLocal/Src_Main.cc ];then
    cp $dirRemote/Src_Main.cc $dirLocal/Src_Main.cc
    echo " copy Src_Main.cc ..."
    echo " copy Src_Main.cc ..." >> $dirLocal/Update.log
else
    UpdateFlag=`diff $dirRemote/Src_Main.cc $dirLocal/Src_Main.cc | wc -l`
    if [ $UpdateFlag -gt 0 ];then
        cp $dirRemote/Src_Main.cc $dirLocal/Src_Main.cc
        echo " Update Src_Main.cc ..."
        echo " Update Src_Main.cc ..." >> $dirLocal/Update.log 
    fi
fi
cp $dirRemote/Src_TSMap $dirLocal
cp $dirRemote/Src_TSMap.cc $dirLocal
cp $dirRemote/Src_Main $dirLocal
cp $dirRemote/Src_GetSBP $dirLocal
cp $dirRemote/Src_GetSBP.cc $dirLocal
cp $dirRemote/Src_Convo_Template.cc $dirLocal
cp $dirRemote/Src_Convo_Template $dirLocal
cp $dirRemote/README $dirLocal

for file in $dirRemote/src/*.h $dirRemote/src/*.yaml
do

    arr=(${file//// })
    fname=${arr[$[${#arr[@]}]-1]}
    if [ ! -f $dirLocal/src/$fname ]; then
        mkdir -p $dirLocal/src
        cp $dirRemote/src/$fname $dirLocal/src/$fname 
        echo " Copy src/$fname ... " 
        echo " Copy src/$fname ... " >> $dirLocal/Update.log
    else
        UpdateFlag=`diff $dirRemote/src/$fname $dirLocal/src/$fname | wc -l`
        if [ $UpdateFlag -gt 0 ]; then
            cp $dirRemote/src/$fname $dirLocal/src/$fname
            echo " Update src/$fname ... "
            echo " Update src/$fname ... " >> $dirLocal/Update.log
        fi
    fi

done

if [ ! -d $dirLocal/src/basic ];then
    cp -r $dirRemote/src/basic $dirLocal/src
fi

if [ ! -d $dirLocal/src/Catalog ];then
    cp -r $dirRemote/src/Catalog $dirLocal/src
fi

for file in $dirRemote/src/Plugin*/*.h
do

    arr=(${file//// })
    fname=${arr[$[${#arr[@]}]-1]}
    Plug=${arr[$[${#arr[@]}]-2]}
    if [ ! -f $dirLocal/src/$Plug/$fname ]; then
        mkdir -p $dirLocal/src/$Plug
        cp $dirRemote/src/$Plug/$fname $dirLocal/src/$Plug/$fname 
        echo " Copy src/$Plug/$fname ... " 
        echo " Copy src/$Plug/$fname ... " >> $dirLocal/Update.log
    else
        UpdateFlag=`diff $dirRemote/src/$Plug/$fname $dirLocal/src/$Plug/$fname | wc -l`
        if [ $UpdateFlag -gt 0 ]; then
            cp $dirRemote/src/$Plug/$fname $dirLocal/src/$Plug/$fname
            echo " Update src/$Plug/$fname ... "
            echo " Update src/$Plug/$fname ... " >> $dirLocal/Update.log
        fi
    fi

done


for file in $dirRemote/config/Data/KM2A/*.yaml
do

    arr=(${file//// })
    fname=${arr[$[${#arr[@]}]-1]}
    if [ ! -f $dirLocal/config/Data/KM2A/$fname ]; then
        mkdir -p $dirLocal/config/Data/KM2A
        cp $dirRemote/config/Data/KM2A/$fname $dirLocal/config/Data/KM2A/$fname 
        echo " Copy config/Data/KM2A/$fname ... " 
        echo " Copy config/Data/KM2A/$fname ... " >> $dirLocal/Update.log
    else
        UpdateFlag=`diff $dirRemote/config/Data/KM2A/$fname $dirLocal/config/Data/KM2A/$fname | wc -l`
        if [ $UpdateFlag -gt 0 ]; then
            cp $dirRemote/config/Data/KM2A/$fname $dirLocal/config/Data/KM2A/$fname
            echo " Update config/Data/KM2A/$fname ... "
            echo " Update config/Data/KM2A/$fname ... " >> $dirLocal/Update.log
        fi
    fi

done


for file in $dirRemote/config/Data/WCDA/Mk/*.yaml
do

    arr=(${file//// })
    fname=${arr[$[${#arr[@]}]-1]}
    if [ ! -f $dirLocal/config/Data/WCDA/Mk/$fname ]; then
        mkdir -p $dirLocal/config/Data/WCDA/Mk
        cp $dirRemote/config/Data/WCDA/Mk/$fname $dirLocal/config/Data/WCDA/Mk/$fname 
        echo " Copy config/Data/WCDA/Mk/$fname ... " 
        echo " Copy config/Data/WCDA/Mk/$fname ... " >> $dirLocal/Update.log
    else
        UpdateFlag=`diff $dirRemote/config/Data/WCDA/Mk/$fname $dirLocal/config/Data/WCDA/Mk/$fname | wc -l`
        if [ $UpdateFlag -gt 0 ]; then
            cp $dirRemote/config/Data/WCDA/Mk/$fname $dirLocal/config/Data/WCDA/Mk/$fname
            echo " Update config/Data/WCDA/Mk/$fname ... "
            echo " Update config/Data/WCDA/Mk/$fname ... " >> $dirLocal/Update.log
        fi
    fi

done

for file in $dirRemote/config/Data/WCDA/Cod/*.yaml
do

    arr=(${file//// })
    fname=${arr[$[${#arr[@]}]-1]}
    if [ ! -f $dirLocal/config/Data/WCDA/Cod/$fname ]; then
        mkdir -p $dirLocal/config/Data/WCDA/Cod
        cp $dirRemote/config/Data/WCDA/Cod/$fname $dirLocal/config/Data/WCDA/Cod/$fname 
        echo " Copy config/Data/WCDA/Cod/$fname ... " 
        echo " Copy config/Data/WCDA/Cod/$fname ... " >> $dirLocal/Update.log
    else
        UpdateFlag=`diff $dirRemote/config/Data/WCDA/Cod/$fname $dirLocal/config/Data/WCDA/Cod/$fname | wc -l`
        if [ $UpdateFlag -gt 0 ]; then
            cp $dirRemote/config/Data/WCDA/Cod/$fname $dirLocal/config/Data/WCDA/Cod/$fname
            echo " Update config/Data/WCDA/Cod/$fname ... "
            echo " Update config/Data/WCDA/Cod/$fname ... " >> $dirLocal/Update.log
        fi
    fi

done


if [ ! -d $dirLocal/config/Tutorial ];then
    cp -r $dirRemote/config/Tutorial $dirLocal/config
fi


if [ ! -d $dirLocal/config/Testrun_Crab ];then
    cp -r $dirRemote/config/Testrun_Crab $dirLocal/config
fi


if [ ! -d $dirLocal/Tools ];then
    cp -r $dirRemote/Tools $dirLocal
fi


if [ ! -d $dirLocal/Results ];then
    cp -r $dirRemote/Results $dirLocal
fi
