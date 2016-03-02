#!/bin/bash
infile=$1
outfile=$2

echo "read $1"
echo "write $2"

cat $1|sed "s/><article/>\n<article/g"|sed "s/><inproceedings/>\n<inproceedings/g"|sed "s/><proceedings/>\n<proceedings/g"|sed "s/><incollection/>\n<incollection/g"|sed "s/><book/>\n<book/g">$2
