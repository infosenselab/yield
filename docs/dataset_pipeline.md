# Dataset Pipeline

The dataset is generated as follows using the `/yield_v1/dataset/` scripts:

1.  Retrieve the raw collections using the scripts in the `/yield_v1/dataset/0_dialogue_getters/` subfolder. Results are stored in the `<data_folder>/raw_data/` folder. Some collections are manually collected.
2.  Using the `/yield_v1/dataset/1_raw_to_txt/` subfolder, convert raw collections in into `.txt` files. These are stored in the `<data_folder>/intermediate_data/01_txt_files/` folder.
3.  The contents of `<data_folder>/intermediate_data/01_txt_files/` are **manually copied** to `<data_folder>/intermediate_data/02_manual_tagging/`, where any remaining transformations not covered by earlier code are done manually. NOTE: There is no `/yield_v1/dataset/` script numbered `2_`. This is done so the logic matches the order in the data folder.
4.  The processed `.txt` files are converted to `.json` using the `/yield_v1/dataset/3_tagged_to_json/intermediate_manual_docs.ipynb` script. These are stored in `<data_folder>/intermediate_data/03_json_conversion/`.
5.  The `.json` files are then fully standardized using the files in the `/yield_v1/dataset/4_dialogue_standardizer/` subfolder. They are stored in the `<data_folder>/yield_base/` folder. These files are now formatted in the dataset schema.
6.  The `<data_folder>/yield_base/` folder contains the full dataset, but in individual files. The `/yield_v1/dataset/5_split/train_dev_test_split.ipynb` script splits the files into training, development and testing sets. The split dataset is in `<data_folder>/yield_v1/`. Indexes of the dataset are also generated in this step and stored in the `<data_folder>/indexes/` folder.
7.  To obtain the fine-tuning dataset, use the `/yield_v1/dataset/6_finetuning/finetuning_formatting.ipynb` script. The result is stored in `<data_folder>/yield_v1_finetuning/`.
