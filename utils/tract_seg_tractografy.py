import os


class TractGrafySeg:
    def __init__(self):
        self.sh2peaks_command = ("sh2peaks -nthreads {} {}/WM_FODs_normalised.nii.gz"
                                 " {}/WM_FODs_peaks.nii.gz -force")

        self.sh2peaks_command_lardi = ("sh2peaks -nthreads {} {}/normalized_WM_FODs.nii.gz"
                                       " {}/WM_FODs_peaks.nii.gz -force")

        self.sh2peaks_evaluate_command = ("sh2peaks -nthreads {} {}/raw_{}.nii.gz"
                                          " {}/WM_FODs_peaks.nii.gz -force")

        # self.tractseg_command = ("docker run -v {}/HARDI_data/:/data -t wasserth/tractseg_container:master TractSeg"
        self.tractseg_command = ("docker run -v {}/:/data -t wasserth/tractseg_container:master TractSeg"
                                 " -i /data/WM_FODs_peaks.nii.gz --output_type tract_segmentation")
        # " --bvals /data/bvals --bvecs /data/bvecs")

        # self.ending_command = ("docker run -v {}/HARDI_data/:/data -t wasserth/tractseg_container:master TractSeg"
        self.ending_command = ("docker run -v {}/:/data -t wasserth/tractseg_container:master TractSeg"
                               " -i /data/WM_FODs_peaks.nii.gz --output_type endings_segmentation")

        # self.tom_command = ("docker run -v {}/HARDI_data/:/data -t wasserth/tractseg_container:master TractSeg"
        self.tom_command = ("docker run -v {}/:/data -t wasserth/tractseg_container:master TractSeg"
                            " -i /data/WM_FODs_peaks.nii.gz --output_type TOM")

        # self.tracking_command = ("docker run -v {}/HARDI_data/:/data -t wasserth/tractseg_container:master Tracking"
        self.tracking_command = ("docker run -v {}/:/data -t wasserth/tractseg_container:master Tracking"
                                 " -i /data/WM_FODs_peaks.nii.gz")

    def segmented_tracts(self, id: str, id_dir: str, n_threads: int):
        # sh2peaks_cmd = self.sh2peaks_command.format(n_threads, id_dir, id_dir)
        tractseg_cmd = self.tractseg_command.format(id_dir, id)

        # os.system(sh2peaks_cmd)

        os.system(tractseg_cmd)

    def tractography_test(self, id: str, id_dir: str, n_threads: int):
        sh2peaks_cmd = self.sh2peaks_command.format(n_threads, id_dir, id_dir)
        tractseg_cmd = self.tractseg_command.format(id_dir, id)
        ending_command = self.ending_command.format(id_dir, id)
        tom_command = self.tom_command.format(id_dir, id)
        tracking_command = self.tracking_command.format(id_dir, id)

        os.system(sh2peaks_cmd)

        os.system(tractseg_cmd)
        os.system(ending_command)
        os.system(tom_command)
        os.system(tracking_command)

    def tractography_test_evaluate(self, id: str, id_dir: str, n_threads: int):
        sh2peaks_cmd = self.sh2peaks_evaluate_command.format(n_threads, id_dir, id, id_dir)
        tractseg_cmd = self.tractseg_command.format(id_dir, id)
        ending_command = self.ending_command.format(id_dir, id)
        tom_command = self.tom_command.format(id_dir, id)
        tracking_command = self.tracking_command.format(id_dir, id)

        os.system(sh2peaks_cmd)

        os.system(tractseg_cmd)
        os.system(ending_command)
        os.system(tom_command)
        os.system(tracking_command)

    def tractography_test_lardi(self, id: str, id_dir: str, n_threads: int):
        sh2peaks_cmd = self.sh2peaks_command_lardi.format(n_threads, id_dir, id_dir)
        tractseg_cmd = self.tractseg_command.format(id_dir, id)
        ending_command = self.ending_command.format(id_dir, id)
        tom_command = self.tom_command.format(id_dir, id)
        tracking_command = self.tracking_command.format(id_dir, id)

        os.system(sh2peaks_cmd)

        os.system(tractseg_cmd)
        os.system(ending_command)
        os.system(tom_command)
        os.system(tracking_command)

    def build_mask(self,
                   id,
                   n_threads,
                   base_dir="generate_data_model",
                   type_directions="HARDI_data"):
        id_dir = f"{base_dir}/{id}/{type_directions}"

        if type_directions == "":
            self.tractography_test_evaluate(id, id_dir, n_threads=n_threads)

        if type_directions == "HARDI_data":
            self.tractography_test(id, id_dir, n_threads=n_threads)

        if type_directions == "LARDI_data_2":
            self.tractography_test_lardi(id, id_dir, n_threads=n_threads)

if __name__ == "__main__":
    tract_seg = TractGrafySeg()
    tract_seg.build_mask("100610", 20, "/home/mateus/Downloads")
