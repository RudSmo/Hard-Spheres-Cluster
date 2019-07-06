#/bin/bash

for k in {1..300..5}; do
  mkdir data/v3/N${k};
  cp source/src.py data/v3/N${k}/;
  cp source/statistics.py data/v3/N${k}/;
  cp source/bootstrap.py data/v3/N${k}/;
  cd data/v3/N${k}/;
   sed "s/^NPar=.*$/NPar=${k}/g" src.py > src${k}.py;
   rm src.py;
   for j in {1..100}; do
    mkdir v$j/;
    cp src${k}.py v$j/;
    python3 src${k}.py;
    wait;
   done;
   rm -rf v* ; 
   python3 bootstrap.py;
   wait;
   sed "s/^NPar=.*$/NPar=${k}/g" statistics.py > statistics${k}.py
   rm statistics.py
   python3 statistics${k}.py
  wait;
  paste avg.txt >> ../outfile.txt
 cd ../;
 cd ../;
 cd ../;
 wait;
done
	
