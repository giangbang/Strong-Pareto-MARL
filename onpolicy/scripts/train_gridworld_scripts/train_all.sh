# #!/bin/sh

# for f in *.sh; do
#     echo "$f" 
#     if [ "$f" = "train_all.sh" ] || [![ $f == *"ns"* ]]; then
#         echo ""
#     else
#         echo "$f" 
#     fi;
# done

./train_plan4_ns_mgda.sh
./train_plan4_ns_mgdapp.sh
./train_plan4_ns_mappo.sh
