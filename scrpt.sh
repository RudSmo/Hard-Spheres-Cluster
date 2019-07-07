#/bin/bash

for k in {2..55}; do
  mkdir data/v1/N${k};
  cp source1/src.py data/v1/N${k}/;
  cp source1/statistics.py data/v1/N${k}/;
  cp source1/bootstrap.py data/v1/N${k}/;
  cd data/v1/N${k}/;
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
  paste clstr_re_avg.txt >> ../clstr_re_outfile.txt
  paste acc_re_avg.txt >> ../acc_re_outfile.txt
  paste pcl_re_avg.txt >> ../pcl_re_outfile.txt
  paste clstr_avg.txt >> ../clstr_outfile.txt
  paste acc_avg.txt >> ../acc_outfile.txt
  paste pcl_avg.txt >> ../pcl_outfile.txt
 cd ../;
 cd ../;
 cd ../;
 wait;
done
	
