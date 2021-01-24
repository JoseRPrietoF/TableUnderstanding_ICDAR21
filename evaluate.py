from utils.metrics import evaluate_graph_IoU, read_results
import os, glob

def read_list_files(p, path_files):
    files = glob.glob(os.path.join(path_files, "*pkl"))
    f = open(p, "r")
    l = f.readlines()
    f.close()
    lines = [x.strip().split(".")[0] for x in l]
    res = []
    for line in lines:
        for f in files:
            if line in f:
                res.append(f)
    return res

def main(path_files, name_fold, fresults, type_="col", conjugate=False, all_edges = True):
    """test_dataset.ids, labels, predictions"""
    list_te = read_list_files(name_fold, path_files)
    print(len(list_te))
    min_w = 0.5
    fP, fR, fF, res = evaluate_graph_IoU(list_te, fresults, min_w=min_w, th=0.8, type_=type_, conjugate=conjugate, all_edges=all_edges)
    fP_1, fR_1, fF_1, res_1 = evaluate_graph_IoU(list_te, fresults, min_w=min_w, th=1.0, type_=type_, conjugate=conjugate, all_edges=all_edges)
    print("#####   IoU and alignment of connected components  #####")
    print("Mean Precision IoU th 0.8 on test : {}".format(fP))
    print("Mean Recal IoU th 0.8 on test : {}".format(fR))
    print("Mean F1 IoU th 0.8 on test : {}".format(fF))
    print("Mean Precision IoU th 1.0 on test : {}".format(fP_1))
    print("Mean Recal IoU th 1.0 on test : {}".format(fR_1))
    print("Mean F1 IoU th 1.0 on test : {}".format(fF_1))

if __name__ == "__main__":
    conjugate=True
    all_edges=False
    type_="row"
    # corpus = "icdar_abp_1098"
    corpus = "icdar_488"
    # corpus = "icdar19_abp_small"
    # path_files = "/data/READ_ABP_TABLE_ICDAR/icdar_488/all/graphs_preprocessed/graphs_structure_BL_lda_20topics_maxBLWidth1.5_mincell0.1_multpunct2/"
    name_fold = "/data2/jose/corpus/tablas_DU/{}/fold1.fold".format(corpus)
    path_files = "/data/READ_ABP_TABLE_ICDAR/icdar_488/all/graphs_structure_BL_maxBLWidth1.5_multpunct_w4_h1_mult_side_h0_w0_bound0_rows9/"
    fresults = "/data/READ_ABP_TABLE_ICDAR/icdar_488/all/graphs_structure_BL_maxBLWidth1.5_multpunct_w4_h1_mult_side_h0_w0_bound0_rows9/results_rows.txt"
    # path_files = "/data/READ_ABP_TABLE_ICDAR/icdar_488/all/graphs_structure_BL_maxBLWidth1.5_multpunct_w1_h1_mult_side_h0_w0_bound0_cols13/"
    # fresults = "/data/READ_ABP_TABLE_ICDAR/icdar_488/all/graphs_structure_BL_maxBLWidth1.5_multpunct_w1_h1_mult_side_h0_w0_bound0_cols13/results_cols.txt"

    # fresults="works_icdar488/works_conjugate_COL/work_graph_COL_BL_NLL_64,64,64,64,32,32,32,16,16ngfs_base_1_notext_maxBLWidth1.5_mincell0.1_multpunct_w0.5_h5_mult_side_h0_w0_bound0_cols10/results.txt"

    # path_files = "/data/READ_ABP_TABLE_ICDAR/icdar_488/all/graphs_structure_BL_maxBLWidth1.5_mincell0.1_multpunct2/"
    # fresults="/data/READ_ABP_TABLE_ICDAR/icdar_488/all/graphs_structure_BL_maxBLWidth1.5_mincell0.1_multpunct2/results.txt"

    # fresults = "/data2/jose/projects/TableUnderstanding/works_icdar488/works_GL_COL/work_prueba_ngf_64,64,32,32,16__maxBLWidth1.5_mincell0.1_multpunct2_GL/results.txt"
    # fresults = "/data2/jose/projects/TableUnderstanding/works_icdar488/works_conjugate_COL/work_graph_COL_BL_64,32,32,32,32,32,32,16,16ngfs_base_1_lda_maxBLWidth1.5_mincell0.1_multpunct2/results.txt"
    # fresults = "/data2/jose/projects/TableUnderstanding/works_icdar488/works_conjugate_ROW/work_graph_ROW_BL_64,32,16ngfs_base_1_lda_20topics_maxBLWidth1.5_mincell0.1_multpunct2/results.txt"
    # path_files = "/data/READ_ABP_TABLE_ICDAR/icdar_488/all/graphs_structure_BL_maxBLWidth1.5_mincell0.1_multpunct_w1_h1_mult_side_h0_w0_bound0_cols/"
    # fresults = "/data/READ_ABP_TABLE_ICDAR/icdar_488/all/graphs_structure_BL_maxBLWidth1.5_mincell0.1_multpunct_w1_h1_mult_side_h0_w0_bound0_cols/results.txt"
    main(path_files, name_fold, fresults, type_, conjugate=conjugate, all_edges=all_edges)
