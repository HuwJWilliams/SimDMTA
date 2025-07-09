#! /bin/bash

# Run full GNINA docking workflow for molecule passed as command line argument

# Run script as:
# ./dock_gnina <SMILES> <molecule name> <N CPUs>

# Location of GNINA binary:
gnina_bin="/users/yhb18174/Recreating_DMTA/docking/gnina"

# Path of directory containing scripts:
#scripts_path=$(dirname $(realpath $0))
scripts_path="/users/xpb20111/programs/docking/gnina"

# Get SMILES and molecule name from the command line:
smi=$1
molname=$2
n_cpus=$3

# Set receptor file and docking box (to replicate FRED docking):
receptor_pdb="${scripts_path}/receptors/4bw1_5_conserved_HOH.pdbqt"
docking_box_centre=\
"--center_x 14.66 \
--center_y 3.41 \
--center_z 10.47"
docking_box_size=\
"--size_x 17.67 \
--size_y 17 \
--size_z 13.67"

# Make a new directory for each molecule:
#mkdir ${molname}
#cd ${molname}

# Generate conformer using OpenBabel:
#obabel -:"${smi}" -O ${molname}.pdb --gen3d

# Generate conformers using OpenEye, ignoring torsion angles which are searched within GNINA:
${scripts_path}/generate_conformers_openeye.py "${smi}" "${molname}" --No_TorDrive ${docking_box_centre} | tee confomer_stats.dat
#conformer_stats=`../generate_conformers_openeye.py "${smi}" "${molname}" false true false "[14.66, 3.41, 10.47]"`
#echo "${conformer_stats}" |& tee confomer_stats.dat

# Split sdf file into separate molecules:
obabel -isdf ${molname}_confs.sdf -O ${molname}_conf.sdf -m

# Save time taken:
echo "molname,conf_no,CNN_usage,time" >> timings.csv

# Do docking and score new poses with different CNN involvement:

for cnn_usage in "none" #"rescore" #"refinement"
do

for ligand_sdf_file in `ls -tr ${molname}_conf*.sdf | grep "${molname}_conf[0-9]\+.sdf"`
do

echo ""
echo "Docking conformer: ${ligand_sdf_file}"
echo "With CNN usage: ${cnn_usage}"

ligand_filename=${ligand_sdf_file%.*}
conf_n=${ligand_filename##*conf}

start_t="$(date -u +%s)"

# Run GNINA docking:
${gnina_bin} \
--receptor ${receptor_pdb} \
--ligand ${ligand_sdf_file} \
--out ${ligand_filename}_gnina_docked_${cnn_usage}CNN.sdf \
--log ${ligand_filename}_gnina_docked_${cnn_usage}CNN.log \
--atom_terms ${ligand_filename}_gnina_docked_atomistic_${cnn_usage}CNN.log \
${docking_box_centre} \
${docking_box_size} \
--exhaustiveness 8 \
--num_modes 9 \
--cpu ${n_cpus} \
--no_gpu \
--addH 0 \
--stripH 0 \
--seed 13 \
--cnn_scoring ${cnn_usage}
# Save stdout and stderr to files:
# >> >(tee stdout.dat) 2>> stderr.dat
# Don't need autobox ligand as docking box position is given explicitly:
#--autobox_ligand /users/xpb20111/pIC50/docking/receptors/4bw1_5_conserved_RF-Score.pdb \
# Change pose sorting:
#--pose_sort_order CNNscore, CNNaffinity, Energy

end_t="$(date -u +%s)"

elapsed_t="$(($end_t-$start_t))"

echo "Time taken for docking: ${elapsed_t} s"

echo "${molname},${conf_n},${cnn_usage},${elapsed_t}" >> timings.csv

# Process GNINA output files into csv:
${scripts_path}/conv_gnina_log_to_csv.py ${ligand_filename}_gnina_docked_${cnn_usage}CNN.log --molname ${molname} --conf ${conf_n} --cnn ${cnn_usage}

done

done

# Combine all csv files:
${scripts_path}/cat_csv.py `ls -tr *_gnina_docked_*CNN.csv` > ${molname}_all_gnina_data.csv

## Save best docking score:
#${scripts_path}/get_best_score.py ${molname}_all_gnina_data.csv > ${molname}_best_docking_score.txt

## Clean up files into tar:
tar -czvf ${molname}_gnina_logs_and_confs.tar.gz ${molname}_conf*_gnina_docked_*.log ${molname}_conf*.sdf ${molname}.sdf ${molname}_conf*.csv stdout.dat stderr.dat --remove-files

cd ../

#tar -czvf ${molname}.tar.gz ${molname} --remove-files

