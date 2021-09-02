#!/usr/bin/env bash
loss=NLL
ngf=64,64,64,64,32,32,32,16,16 
epoch=600
conjugate=COL
model=EdgeConv
data_path=/data2/jose/projects/TableUnderstanding_ICDAR21/data_created/data_created_preprocessed
work_path=work_icdar19_abp_small_graph_${conjugate}_BL_${loss}_${ngf}ngfs_base_notext_original_${model}
python3.6 main_graph.py --batch_size 24 \
--data_path ${data_path} \
--epochs ${epoch} \
--work_dir ${work_path} \
--fold_paths folds_ICDAR21_TableUnderstanding/icdar19_abp_small/ \
--layers ${ngf} --adam_lr 0.001 --conjugate ${conjugate} --show_test 300 --load_model True --model ${model}