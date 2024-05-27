import os
from tqdm import tqdm


def run_processing(id_dir, n_threads):
    sh2peaks_command_hardi = "sh2peaks -nthreads {} {}/LARDI_data/WM_FODs.nii.gz {}/LARDI_data/WM_FODs_peaks.nii.gz -force"
    sh2peaks_command_lardi = "sh2peaks -nthreads {} {}/HARDI_data/WM_FODs.nii.gz {}/HARDI_data/WM_FODs_peaks.nii.gz -force"
    tractseg_command = "docker run -v {}/HARDI_data:/data -t wasserth/tractseg_container:master TractSeg -i /data/WM_FODs_peaks.nii.gz --output_type tract_segmentation --bvals /data/bvals --bvecs /data/bvecs"
    tractseg_lardi_command = "docker run -v {}/LARDI_data:/data -t wasserth/tractseg_container:master TractSeg -i /data/WM_FODs_peaks.nii.gz --output_type tract_segmentation --bvals /data/data_b1000_g32_bvals --bvecs /data/data_b1000_g32_bvecs"

    sh2peaks_cmd_hardi = sh2peaks_command_hardi.format(n_threads, id_dir, id_dir)
    sh2peaks_cmd_lardi = sh2peaks_command_lardi.format(n_threads, id_dir, id_dir)
    tractseg_cmd = tractseg_command.format(id_dir)
    tractseg_lardi_cmd = tractseg_lardi_command.format(id_dir)

    # Execute os comandos
    os.system(sh2peaks_cmd_hardi)
    os.system(sh2peaks_cmd_lardi)

    os.system(tractseg_cmd)
    os.system(tractseg_lardi_cmd)


def main(base_dir, n_threads):
    # Lista de IDs
    ids = os.listdir(base_dir)

    # Barra de progresso personalizada com tqdm
    progress_bar = tqdm(ids, desc="Processando IDs", unit="ID")

    # Itera sobre os IDs
    for id in progress_bar:
        id_dir = os.path.join(base_dir, id)
        if os.path.isdir(id_dir):
            run_processing(id_dir, n_threads=n_threads)
            progress_bar.set_description(f"Processamento concluído para ID: {id}")


if __name__ == "__main__":
    base_dir = '/mnt/datahdd/dataset_organized/train'
    main(base_dir, 28)
