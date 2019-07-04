#/bin/bash

for k in {10..60..50}; do
  mkdir N${k};
  cp src.py N${k}/;
  cp statistics.py N${k}/;
  cp bootstrap.py N${k}/;
  cd N${k}/;
   sed "s/^NPar=.*$/NPar=${k}/g" src.py > src${k}.py;
   rm src.py;
   for j in {1..100}; do
    mkdir $j/;
    cp src${k}.py $j/;
    python3 src${k}.py;
    wait;
   done;
   python3 bootstrap.py;
   wait;
   sed "s/^NPar=.*$/NPar=${k}/g" statistics.py > statistics${k}.py
   rm statistics.py
   python3 statistics${k}.py
  wait;
  paste avg.txt >> ../outfile.txt
 cd ../;
 wait;
done
	
