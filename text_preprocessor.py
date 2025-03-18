import os
from datasets import load_dataset
OUTPUT_FILE_TEMPLATE = "shared-file-storage/preprocessed_data/preprocessed_{rank}.txt"

def preprocess_text(text):
    return text.lower().strip().split()

def load_ds(rank, total_procs):
    # TODO: There should be some logic about loading the dataset here. As we only have 2 Gigabytes of memory available per process, maybe that should play a role as well ;)
    dataset=load_dataset("allenai/c4", "multilingual", split="train", streaming=True)
    print(dataset.shard(num_shards=total_procs, index=rank))

    for i, data in enumerate(dataset.shard(num_shards=total_procs, index=rank)):
        if i>=500: #For running on my laptop with small sample size 
            break
        yield data['text']
    return data
    #raise NotImplementedError("This function has not been implemented yet.")

def write_preprocessed_text(preprocessed_text, rank):
    #Create directory setup if not available.
    os.makedirs(os.path.dirname(OUTPUT_FILE_TEMPLATE), exist_ok=True)

    with open(OUTPUT_FILE_TEMPLATE.format(rank=rank), "w") as f:
        for line in preprocessed_text:
            f.write("\t".join(line))
            f.write("\n")

def main():
    # TODO: Here you should try to check the rank of this process and the total number of processes that are spawned
    local_rank = int(os.getenv("PROC_RANK", 0)) # TODO: This should be the rank of the process
    total_procs = int(os.getenv("TOTAL_PROCS", 1)) # TODO: This should be the total number of processes that are spawned
    # Extract the text to process
    print("Process Rank: ", local_rank, ", Total Processes: ", total_procs)
    text_to_process = load_ds(local_rank, total_procs) # TODO: pass the relevant arguments (if any)
    # Preprocess the text
    preprocessed_text = [preprocess_text(text) for text in text_to_process]
    # TODO: Write this somewhere
    write_preprocessed_text(preprocessed_text, rank=local_rank)
    print("Process ", local_rank, " Completed. Processed", len(preprocessed_text), " samples." )
    #raise NotImplementedError("This function has not been fully implemented yet.")

if __name__ == "__main__":
    main()
