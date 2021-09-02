data_created=/data2/jose/projects/TableUnderstanding_ICDAR21/data_created
data_xml=~/corpus/tablas_DU/icdar19_abp_small/
data_preprocessed=/data2/jose/projects/TableUnderstanding_ICDAR21/data_created/data_created_preprocessed
cd data
# create_graphs.py data_path(PAGE-XML) dir_dest min_cell min_num_neighrbors weight_radius_w weight_radius_h j_h h_w
python3.6 create_graphs.py ${data_xml} ${data_created} 0 1 1 1 1 1
python3.6 preprocess.py ${data_created} ${data_preprocessed}
cd ..