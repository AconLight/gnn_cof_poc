import subprocess

import numpy as np

if __name__ == "__main__":
    datasets = ['HRSS', 'MI-V', 'MI-F', 'WBC','ANNT', 'THYR', 'MUSK', 'MAMO', 'VERT', 'WINE',
                'BREAST', 'PIMA', 'GLASS', 'MNIST', 'SPEECH', 'SAT', 'PEN', 'OPT', 'SHUTTLE', 'ARR']

    datasets = ['WBC', 'ARR', 'SAT', 'OPT']
    # datasets = ['SAT', 'OPT']
    datasets = ['SPEECH']

    # samples = "MIXED-1_MIXED-5_MIXED-10_MIXED-15"
    # samples = "MIXED-1_MIXED-2"
    samples = "MIXED-2"
    # samples = "MIXEDHIGHDIM-2"

    for proportion in [1]:#np.arange(0.05, 0.5, 0.05):
        for dataset in datasets:
            subprocess.run(["python", "./src/run/main.py",
                            f"--samples={samples}",
                            f"--datasets={dataset}",
                            f"--start_k={3}",
                            f"--max_k={3}",
                            f"--proportion={proportion}",
            ])


    # datasets = ['HRSS', 'MI-V', 'MI-F', 'WBC','ANNT', 'THYR', 'MUSK', 'MAMO', 'VERT', 'WINE',
    #             'BREAST', 'PIMA', 'GLASS', 'MNIST', 'SPEECH', 'SAT', 'PEN', 'OPT', 'SHUTTLE', 'ARR']
