# Extract
**scripts/extract.py**
- `from mydatasets.base_dataset import BaseDataset`
- For each page
    - `BaseDataset._extract_content` using `pymupdf get_pixmap` to get page screenshot
    - `BaseDataset._extract_content` using `pymupdf get_text` to get text
    - Save image page on 'tmp/{dataset_name}/{doc_id}_{page}.png'
    - Save text page on 'tmp/{dataset_name}/{doc_id}_{page}.txt'

# Retrieve
**scripts/retrieve.py**
- `base.yaml:defaults:retrieval` decide which text/image retrieval to index on run
- For text retrieval:
  - Instantiate `retrieval.text_retrieval.ColbertRetrieval`
  - Load text embeddings from `./tmp/${retrieval.model_name}/${retrieval.text_question_key}`
  - Store results with key `text-top-${retrieval.top_k}-${retrieval.text_question_key}`
- For image retrieval:
  - Instantiate `retrieval.image_retrieval.ColpaliRetrieval`
  - Load image embeddings from `./tmp/${retrieval.model_name}/${retrieval.image_question_key}`
  - Store results with key `image-top-${retrieval.top_k}-${retrieval.image_question_key}`
- For each sample in dataset:
  - Find top-k similar items using the embeddings
  - Save updated samples to `sample-with-retrieval-results.json`

# Dataset
data/FetaTab/samples.json   1016
data/LongDocURL/samples.json   2325
data/MMLongBench/samples.json   1073
data/PaperTab/samples.json   393
data/PaperText/samples.json   2804