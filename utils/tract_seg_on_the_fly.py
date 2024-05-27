import os


class TractSeg:
    def __init__(self):
        self.sh2peaks_command = "sh2peaks -nthreads {} {}/tmp_wm_{}.nii.gz {}/WM_FODs_peaks.nii.gz -force"

        self.tractseg_command = ("docker run -v {}:/data -t wasserth/tractseg_container:master TractSeg"
                                 " -i /data/WM_FODs_peaks.nii.gz --output_type tract_segmentation")
        # " --bvals /data/bvals --bvecs /data/bvecs")

        self.tractseg_lardi_command = (
            "docker run -v {}/:/data -t wasserth/tractseg_container:master TractSeg"
            " -i /data/WM_FODs_peaks.nii.gz --output_type tract_segmentation")
        # " --bvals /data/data_b1000_g32_bvals --bvecs /data/data_b1000_g32_bvecs")

    def segmented_tracts(self, id: str, id_dir: str, n_threads: int):
        sh2peaks_cmd = self.sh2peaks_command.format(n_threads, id_dir, id, id_dir)
        tractseg_cmd = self.tractseg_command.format(id_dir, id)

        os.system(sh2peaks_cmd)

        os.system(tractseg_cmd)

    def build_mask(self,
                   id,
                   n_threads,
                   base_dir="generate_data_model"):
        id_dir = f"{base_dir}/{id}"

        self.segmented_tracts(id, id_dir, n_threads=n_threads)
