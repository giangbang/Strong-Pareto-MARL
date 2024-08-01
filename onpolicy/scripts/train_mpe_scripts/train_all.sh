# #!/bin/sh

# for f in *.sh; do
#     echo "$f" 
#     if [ "$f" = "train_all.sh" ] || [![ $f == *"ns"* ]]; then
#         echo ""
#     else
#         echo "$f" 
#     fi;
# done

./train_mpe_reference_mappo.sh
./train_mpe_reference_mgda.sh
./train_mpe_reference_mgdapp.sh
./train_mpe_spread_mappo.sh
./train_mpe_spread_mgda.sh
./train_mpe_spread_mgdapp.sh
shutdown now
