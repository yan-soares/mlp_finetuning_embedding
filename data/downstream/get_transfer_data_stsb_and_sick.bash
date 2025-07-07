# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# Download and tokenize data with MOSES tokenizer
#

data_path=.
preprocess_exec=./tokenizer.sed

# Get MOSES
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git
SCRIPTS=mosesdecoder/scripts
MTOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LOWER=$SCRIPTS/tokenizer/lowercase.perl

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

PTBTOKENIZER="sed -f tokenizer.sed"

mkdir $data_path

TREC='http://cogcomp.cs.illinois.edu/Data/QA/QC'
#SICK='http://alt.qcri.org/semeval2014/task1/data/uploads'
SICK='https://zenodo.org/records/2787612/files/SICK.zip?download=1'
BINCLASSIF='https://dl.fbaipublicfiles.com/senteval/senteval_data/datasmall_NB_ACL12.zip'
SSTbin='https://raw.githubusercontent.com/PrincetonML/SIF/master/data'
SSTfine='https://raw.githubusercontent.com/AcademiaSinicaNLPLab/sentiment_dataset/master/data/'
STSBenchmark='http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz'
SNLI='https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
MULTINLI='https://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip'
COCO='https://dl.fbaipublicfiles.com/senteval/coco_r101_feat'

# STSBenchmark (http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)

curl -Lo $data_path/Stsbenchmark.tar.gz $STSBenchmark
tar -zxvf $data_path/Stsbenchmark.tar.gz -C $data_path
rm $data_path/Stsbenchmark.tar.gz
mv $data_path/stsbenchmark $data_path/STS/STSBenchmark

for split in train dev test
do
    fname=sts-$split.csv
    fdir=$data_path/STS/STSBenchmark
    cut -f1,2,3,4,5 $fdir/$fname > $fdir/tmp1
    cut -f6 $fdir/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $fdir/tmp2
    cut -f7 $fdir/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $fdir/tmp3
    paste $fdir/tmp1 $fdir/tmp2 $fdir/tmp3 > $fdir/$fname
    rm $fdir/tmp1 $fdir/tmp2 $fdir/tmp3
done


### download SICK
mkdir $data_path/SICK

for split in train trial test_annotated
do
    urlname=$SICK/sick_$split.zip
    curl -Lo $data_path/SICK/sick_$split.zip $urlname
    unzip $data_path/SICK/sick_$split.zip -d $data_path/SICK/
    rm $data_path/SICK/readme.txt
    rm $data_path/SICK/sick_$split.zip
done

for split in train trial test_annotated
do
    fname=$data_path/SICK/SICK_$split.txt
    cut -f1 $fname | sed '1d' > $data_path/SICK/tmp1
    cut -f4,5 $fname | sed '1d' > $data_path/SICK/tmp45
    cut -f2 $fname | sed '1d' | $MTOKENIZER -threads 8 -l en -no-escape > $data_path/SICK/tmp2
    cut -f3 $fname | sed '1d' | $MTOKENIZER -threads 8 -l en -no-escape > $data_path/SICK/tmp3
    head -n 1 $fname > $data_path/SICK/tmp0
    paste $data_path/SICK/tmp1 $data_path/SICK/tmp2 $data_path/SICK/tmp3 $data_path/SICK/tmp45 >> $data_path/SICK/tmp0
    mv $data_path/SICK/tmp0 $fname
    rm $data_path/SICK/tmp*
done

# remove moses folder
rm -rf mosesdecoder
