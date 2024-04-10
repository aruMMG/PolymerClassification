import pandas as pd
import numpy as np
import csv
import math
import os
def calculate_metrics(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    d = np.sum(confusion_matrix, axis=0)
    precision = []
    recall = []
    for i in range(num_classes):
        recall.append(confusion_matrix[:,i][i]/np.sum(confusion_matrix[:,i]))
        precision.append(confusion_matrix[i][i]/np.sum(confusion_matrix[i]))

    precision = np.array(precision)
    recall = np.array(recall)
    assert not np.isnan(recall).any(), f"recall has nan value."
    # recall = np.nan_to_num(recall, nan=0.0)
    p_s, c_s = 0, 0
    for i in range(num_classes):
        if not math.isnan(precision[i]):
            p_s += precision[i]*d[i]
            c_s += d[i]
    weighted_precision = p_s/c_s
    # weighted_precision = np.sum(precision*d)/np.sum(d)
    weighted_recall = np.sum(recall*d)/np.sum(d)
    avarage = np.trace(confusion_matrix)/np.sum(d)
    return precision, recall, weighted_precision, weighted_recall, avarage

def matrixFromCSV(fileName):
    CM = np.genfromtxt(fileName, delimiter=',')
    n_classes = CM.shape[1]
    folds = int(CM.shape[0]/(n_classes))
    p_all, r_all, wp_all, wr_all, a_all = [],[],[],[],[]
    for fold in range(folds):
        CM_fold = CM[fold*n_classes:(fold+1)*n_classes,:]
        precision, recall, weighted_precision, weighted_recall, avarage = calculate_metrics(CM_fold)
        p_all.append(precision)
        r_all.append(recall)
        wp_all.append(weighted_precision)
        wr_all.append(weighted_recall)
        a_all.append(avarage)
    return np.array(p_all), np.array(r_all), np.array(wp_all), np.array(wr_all), np.array(a_all)

def all_results(filename, dir_path):
    p,r,wp,wr,a = matrixFromCSV(f"{dir_path}{filename}.csv")
    column_averages = np.nanmean(p, axis=0)
    p = np.vstack([p, column_averages])
    column_averages = np.nanmean(r, axis=0)
    r = np.vstack([r, column_averages])
    pr = np.concatenate([p, r], axis=1)
    all = np.vstack((wp, wr, a))
    if not os.path.exists(f"{dir_path}metrices"):
        os.mkdir(f"{dir_path}metrices")
    if not os.path.exists(f"{dir_path}metrices/weighted"):
        os.mkdir(f"{dir_path}metrices/weighted")
    if not os.path.exists(f"{dir_path}metrices/classwise"):
        os.mkdir(f"{dir_path}metrices/classwise")
    np.savetxt(f"{dir_path}metrices/classwise/metrices_classwise_{filename}.csv", pr, delimiter=",")
    np.savetxt(f"{dir_path}metrices/weighted/metrices_all_{filename}.csv", all, delimiter=",")

def all_std(filename, dir_path):
    p,r,wp,wr,a = matrixFromCSV(f"{dir_path}{filename}.csv")
    column_averages = np.nanstd(p, axis=0)
    p = np.vstack([p, column_averages])
    column_averages = np.nanstd(r, axis=0)
    r = np.vstack([r, column_averages])
    pr = np.concatenate([p, r], axis=1)
    all = np.vstack((wp, wr, a))
    if not os.path.exists(f"{dir_path}metrices"):
        os.mkdir(f"{dir_path}metrices")
    if not os.path.exists(f"{dir_path}metrices/weighted"):
        os.mkdir(f"{dir_path}metrices/weighted")
    if not os.path.exists(f"{dir_path}metrices/classwise"):
        os.mkdir(f"{dir_path}metrices/classwise")
    np.savetxt(f"{dir_path}metrices/classwise/metrices_classwise_{filename}.csv", pr, delimiter=",")
    np.savetxt(f"{dir_path}metrices/weighted/metrices_all_{filename}.csv", all, delimiter=",")

def create_new_csv(directory, metric=None):
    # Get list of CSV files in the given directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    
    # Create a new CSV file to store the first lines
    with open(f'{directory}_{metric}.csv', 'w', newline='') as new_csv_file:
        csv_writer = csv.writer(new_csv_file)

        
        # Write first rows from each CSV file
        for filename in csv_files:
            with open(os.path.join(directory, filename), 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                if metric=="classwise":
                    for _ in range(10):
                        next(csv_reader)
                elif metric=="p":
                    pass
                elif metric=="r":
                    next(csv_reader)
                    next(csv_reader)
                first_row = next(csv_reader)
                row_to_write = [filename] + first_row  # Add filename as first column
                csv_writer.writerow(row_to_write)



if __name__=="__main__":
    filename = "last_trans_baseline"
    dir_path = "./logFile/EXP6/c/test/"
    p,r,wp,wr,a = matrixFromCSV(f"{dir_path}{filename}.csv")
    print(wp)
    print(wr)
    print(np.std(wp))
    print(np.std(wr))
    print(p.shape)

    # import os
    # filename = "last_trans"
    # dir_path = "./logFile/EXP5/c/test/"
    # all_results(filename, dir_path)
    # filename = "last_trans_baseline"
    # dir_path = "./logFile/EXP5/c/test/"
    # all_results(filename, dir_path)
    # filename = "last_trans_pre"
    # dir_path = "./logFile/EXP5/c/test/"
    # all_results(filename, dir_path)
    # filename = "last_trans_pre_baseline"
    # dir_path = "./logFile/EXP5/c/test/"
    # all_results(filename, dir_path)
    # filename = "last_incep_pre"
    # dir_path = "./logFile/EXP5/c/test/"
    # all_results(filename, dir_path)
    # filename = "last_incep_pre_baseline"
    # dir_path = "./logFile/EXP5/c/test/"
    # all_results(filename, dir_path)
    # filename = "last_incep"
    # dir_path = "./logFile/EXP5/c/test/"
    # all_results(filename, dir_path)
    # filename = "last_incep_baseline"
    # dir_path = "./logFile/EXP5/c/test/"
    # all_results(filename, dir_path)
    # filename = "last_FC_baseline"
    # dir_path = "./logFile/EXP5/c/test/"
    # all_results(filename, dir_path)
    # filename = "last_FC"
    # dir_path = "./logFile/EXP5/c/test/"
    # all_results(filename, dir_path)
    # filename = "last_res_baseline"
    # dir_path = "./logFile/EXP5/c/test/"
    # all_results(filename, dir_path)
    # filename = "last_res"
    # dir_path = "./logFile/EXP5/c/test/"
    # all_results(filename, dir_path)
        
    # ========================================================
    # This is same as the function all_metrices. For any specific remove comment and make change.
    # p,r,wp,wr,a = matrixFromCSV(f"{dir_path}{filename}.csv")
    # column_averages = np.nanmean(p, axis=0)
    # p = np.vstack([p, column_averages])
    # column_averages = np.nanmean(r, axis=0)
    # r = np.vstack([r, column_averages])
    # pr = np.concatenate([p, r], axis=1)
    # all = np.vstack((wp, wr, a))
    # if not os.path.exists(f"{dir_path}metrices"):
    #     os.mkdir(f"{dir_path}metrices")
    # np.savetxt(f"{dir_path}metrices/metrices_classwise_{filename}.csv", pr, delimiter=",")
    # np.savetxt(f"{dir_path}metrices/metrices_all_{filename}.csv", all, delimiter=",")

    # ==========================================================
    # Uncomment to generate a single file with all results for all comaprative results.
    directory = './logFile/EXP5/c/test/metrices/classwise'
    create_new_csv(directory, metric="classwise")
    directory = './logFile/EXP5/c/test/metrices/weighted'
    create_new_csv(directory, metric="p")
    directory = './logFile/EXP5/c/test/metrices/weighted'
    create_new_csv(directory, metric="r")
