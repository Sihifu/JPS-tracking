# Trajectory Reconstruction from Image Sequence

This project reconstructs the trajectory of particles from a given sequence of images. The project is implemented using Jupyter notebooks and involves several steps to ensure accurate tracking and visualization of the particle trajectories.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

1. Clone the repository:
    ```bash
    git clone https://gitlab.lrz.de/ga45bic/jps-tracking-version-2.git
    cd trajectory-reconstruction
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have Jupyter installed:
    ```bash
    pip install jupyterlab
    ```

## Usage

1. **Prepare Video and Images:**
    - Save the video file into the `/Video_Data/` directory.
    - Save the corresponding image frames into a folder within `Image_Data/` (e.g., `Image_Data/folder_name/`).

2. **Run the Jupyter Notebooks in the Following Sequence:**

    - **[segmentation_tracker.ipynb](segmentation_tracker.ipynb)**
        - Check that `number_training_images` and `history` are set correctly.
        - Ensure the background image is clean (this is crucial).
        - Verify that the paths to the video and images are accurate.
        - Ensure the `array stream string` contains the correct image paths in order.
        - Verify that the array shapes match the frames for accurate matching.

    - **[vizualization_center.ipynb](vizualization_center.ipynb)**
        - Adjust input parameters as needed.
        - Review the video to confirm that tracking is accurate.

    - **[save_smoothed_images.ipynb](save_smoothed_images.ipynb)**

    - **[particle_data_structure.ipynb](particle_data_structure.ipynb)**
        - (TODO) Implement the cut radius calculator.

    - **[normal_vector_trajectory.ipynb](normal_vector_trajectory.ipynb)**

    - **[rotation_matrices_filtered.ipynb](rotation_matrices_filtered.ipynb)**

    - **[roll_out.ipynb](roll_out.ipynb)**

    - **[visualization_clustering_ground_trajectory.ipynb](visualization_clustering_ground_trajectory.ipynb)**
        - Run this notebook for each particle to visualize its ground trajectory.


## Features

- **Trajectory Reconstruction**: Track and visualize the trajectory of particles from an image sequence.
- **Customizable Parameters**: Easily adjust tracking parameters to fit different datasets.
- **Visualization Tools**: Various notebooks provided for visualizing the particle tracking and clustering results.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes. Make sure to follow the project's coding standards and add appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please reach out to [Hoang.dang.tun@gmail.com](mailto:Hoang.dang.tun@gmail.com).



