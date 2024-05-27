import numpy as np
import nibabel as nib
from tqdm import tqdm


def extract_patches(image_data, patch_size, stride, output_dir):
    image_shape = image_data.shape
    patch_depth, patch_height, patch_width = patch_size
    stride_depth, stride_height, stride_width = stride

    count = 0

    # More than 6GB USING slide window
    for d in tqdm(range(0, image_shape[0] - patch_depth + 1, stride_depth)):
        for h in range(0, image_shape[1] - patch_height + 1, stride_height):
            for w in range(0, image_shape[2] - patch_width + 1, stride_width):
                patch = image_data[d:d + patch_depth, h:h + patch_height, w:w + patch_width]

                # Save the patch as NIfTI file
                output_filename = f"patch_{d}_{h}_{w}.nii.gz"
                nib.save(nib.Nifti1Image(patch, affine=np.eye(4)), output_dir + "/" + output_filename)

                count += 1

    print(f"patches: {count}")


def extract_patches_mask(image_data, patch_size, margin, matrix_mask, index_length, output_dir):
    print(f"patches: {matrix_mask[0, :].shape[0]}")

    for idx in tqdm(range(index_length), desc="Outer Loop", mininterval=10):
        x = matrix_mask[0, idx]
        y = matrix_mask[1, idx]
        z = matrix_mask[2, idx]

        x_start = x - margin
        x_end = x_start + patch_size

        y_start = y - margin
        y_end = y_start + patch_size

        z_start = z - margin
        z_end = z_start + patch_size

        if x_start < 0 or y_start < 0 or z_start < 0:
            continue

        output_filename = f"patch_{idx}_.nii.gz"
        patch = image_data[x_start:x_end, y_start:y_end, z_start:z_end, :]
        nib.save(nib.Nifti1Image(patch, affine=np.eye(4)), output_dir + "/" + output_filename)


from tqdm import tqdm
import numpy as np
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor


def process_patch(patch_idx, x, y, z, margin, patch_size, image_data, output_dir):
    x_start = x - margin
    x_end = x_start + patch_size

    y_start = y - margin
    y_end = y_start + patch_size

    z_start = z - margin
    z_end = z_start + patch_size

    if x_start < 0 or y_start < 0 or z_start < 0:
        return

    output_filename = f"patch_{patch_idx}_.nii.gz"
    patch = image_data[x_start:x_end, y_start:y_end, z_start:z_end, :]
    nib.save(nib.Nifti1Image(patch, affine=np.eye(4)), output_dir + "/" + output_filename)


def extract_patches_mask_parallel(image_data, patch_size, margin, matrix_mask, index_length, output_dir, num_threads=8):
    print(f"patches: {matrix_mask[0, :].shape[0]}")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []

        for idx in tqdm(range(index_length), desc="Processing Patches", mininterval=10):
            x = matrix_mask[0, idx]
            y = matrix_mask[1, idx]
            z = matrix_mask[2, idx]

            futures.append(executor.submit(process_patch, idx, x, y, z, margin, patch_size, image_data, output_dir))

        # Wait for all futures to complete
        for future in tqdm(futures, desc="Waiting for Completion", mininterval=10):
            future.result()


if __name__ == "__main__":
    # Example image with shape (100, 100, 100)
    image_shape = (10, 10, 10)
    image_data = np.random.rand(*image_shape)  # Generating random data

    # Parameters
    patch_size = (5, 5, 5)  # 3D patch size
    stride = (2, 2, 2)  # Stride for sliding the window
    output_directory = "data_tmp/"  # Output directory to save patches

    import os

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    extract_patches(image_data, patch_size, stride, output_directory)

# for i, idx in enumerate(range(index_length)):
#
#     x = final_index[0, idx]
#     y = final_index[1, idx]
#     z = final_index[2, idx]
#
#     x_start = x - self.margin
#     x_end = x_start + self.size_3d_patch
#
#     y_start = y - self.margin
#     y_end = y_start + self.size_3d_patch
#
#     z_start = z - self.margin
#     z_end = z_start + self.size_3d_patch
#
#     fodlr_3D_patches = fodlr[x_start:x_end, y_start:y_end, z_start:z_end, :]
#
#

# extract_patches(image_data=fodgt, patch_size=(self.size_3d_patch,
        #                                               self.size_3d_patch,
        #                                               self.size_3d_patch),
        #                 stride=None,
        #                 output_dir="utils/data_tmp/")

        # extract_patches_mask_parallel(image_data=fodgt,
        #                               patch_size=self.size_3d_patch,
        #                               margin=self.margin,
        #                               matrix_mask=final_index,
        #                               index_length=index_length,
        #                               output_dir="utils/data_tmp")