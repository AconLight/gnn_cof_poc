import subprocess

if __name__ == "__main__":
    datasets = ['HRSS', 'MI-V', 'MI-F', 'WBC','ANNT', 'THYR', 'MUSK', 'MAMO', 'VERT', 'WINE',
                'BREAST', 'PIMA', 'GLASS', 'MNIST', 'SPEECH', 'SAT', 'PEN', 'OPT', 'SHUTTLE', 'ARR']

    datasets = ['PEN', 'SHUTTLE']
    # ['WBC', 'MUSK', 'ARR', 'SPEECH', 'OPT', 'MNIST'] # high dimensions

    samples = "MIXED-1_MIXED-2_MIXED-3_MIXED-4"

    modes = 'dist dist_angle'
    #['dist', 'angle' 'dist_angle']

    for dataset in datasets:
        subprocess.run(["python", "./src/run/main.py",
                        f"--samples={samples}",
                        f"--datasets={dataset}",
                        f"--start_k={3}",
                        f"--max_k={3}",
                        # f"--modes={modes}",
        ])