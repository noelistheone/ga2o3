#!/bin/bash
# full sputter-film melt-quench for the 4 lab dopants, serial on GPU0
cd /home/lawrence/Physics/Ga2O3-Sandbox
for spec in "Si 0.0234" "Sn 0.0265" "Mg 0.0101" "Zn 0.0263"; do
    set -- $spec
    conda run -n ga2o3 python scripts/sputter_film_structure.py --gpu 0 --dopant $1 --conc $2 \
        --melt_steps 3000 --quench_steps 3000 >> logs/sputter_full.log 2>&1
    echo "[done] $1 $2" >> logs/sputter_full.log
done
echo "ALL SPUTTER DONE" >> logs/sputter_full.log
