from huggingface_hub import snapshot_download
import os

corpus_dir = os.path.abspath("./corpus/textbooks")
print(f"Downloading MedRAG/textbooks to {corpus_dir} ...")
snapshot_download(
    repo_id="MedRAG/textbooks",
    repo_type="dataset",
    local_dir=corpus_dir,
    resume_download=True
)
print("Download finished successfully!")
