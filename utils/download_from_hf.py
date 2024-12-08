"""this script is only for single file download

# for whole dataset use the following command
huggingface-cli download 'Writer/omniact' --local-dir '/home/shaotao/DATA/omniact' --repo-type 'dataset'
"""


from huggingface_hub import hf_hub_download


if __name__ == "__main__":
    repo_id = "Writer/omniact"
    file_path = "file_path"
    cache_dir = '/home/shaotao/DATA/omniact'
    downloaded_file_path = hf_hub_download(repo_id=repo_id, 
                                        filename=file_path,
                                        # subfolder=file_path,
                                        repo_type='dataset',
                                        cache_dir=cache_dir)

    print(f"File downloaded to {downloaded_file_path}")
