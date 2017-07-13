#!/bin/bash
#
# unbundle.sh: use clang clang-offload-bundler tool to unbundle files
#              This script is the companion to bundle.sh.  It will name the 
#              generated files using the conventions used in the coral compiler. 
#
#  Written by Greg Rodgers
# 
HCC2=${HCC2:-/opt/rocm/hcc2}
EXECBIN=$HCC2/bin/clang-offload-bundler
if [ ! -f $EXECBIN ] ; then 
   echo "ERROR: $EXECBIN not found"
   exit 1
fi

infilename=${1:-ll}
if [ -f $infilename ] ; then 
   ftype=${infilename##*.}
   tnames="$infilename"
else
#  Input was not a file, work on all files in the directory of specified type
   ftype=$infilename
   tnames=`ls *.$ftype 2>/dev/null`
   if [ $? != 0 ] ; then 
      echo "ERROR: No files of type $ftype found"
      echo "       Try a filetype other than $ftype"
      exit 1
   fi
fi

for tname in $tnames ; do 
   mname=${tname%.$ftype}
   if [ "$ftype" == "o" ] ; then 
      otargets=`strings $tname | grep "__CLANG_OFFLOAD_BUNDLE" `
      for longtarget in $otargets ; do 
         targets="$targets ${longtarget:24}"
      done
   else
      if [ "$ftype" != "bc" ] ; then 
         targets=`grep "__CLANG_OFFLOAD_BUNDLE____START__" $tname | cut -d" " -f3`
      else
#        Hack to find bc targets from a bundle
         for t in `strings $tname | grep "openmp-" ` ; do 
            t=${t%*k}
            t=${t%*\'>}
            targets="$targets $t"
         done 
         host=`strings $tname | grep "host-" `
         targets="$targets ${host%*BC}"
      fi
   fi
   targetlist=""
   sepchar=""
   fnamelist=""
   for target in $targets ; do 
      targetlist=$targetlist$sepchar$target
      fnamelist=$fnamelist$sepchar${tname}.$target
      sepchar=","
   done
   if [ "$targetlist" != "" ] ; then 
      echo $EXECBIN -unbundle -type=$ftype -inputs=$tname -targets=$targetlist -outputs=$fnamelist
      $EXECBIN -unbundle -type=$ftype -inputs=$tname -targets=$targetlist -outputs=$fnamelist
      if [ $? != 0 ] ; then 
         echo "ERROR: $EXECBIN failed."
         echo "       The failed command was:"
         echo $EXECBIN -unbundle -type=$ftype -inputs=$tname -targets=$targetlist -outputs=$fnamelist
         exit 1
      fi
   fi
done
