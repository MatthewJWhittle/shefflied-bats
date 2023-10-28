from sdm.models import S2Dataset

def main():
    input_vars = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14
        ]
    n_input_bands = len(input_vars)

    dataset = S2Dataset(
        "data/processed/s2-lidar/s2-lidar-stack.tif",
        input_bands=input_vars,
        target_band=[15],
    )


    # Load the first 20 items
    for i in range(0, 20):
        item = dataset[i]

if __name__ == "__main__":
    main()
