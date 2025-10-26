import kagglehub

if __name__ == "__main__":
    kagglehub.login()

    # Download latest version
    path = kagglehub.dataset_download("engeddy/astrophysical-objects-image-dataset")

    print("Path to dataset files:", path)
