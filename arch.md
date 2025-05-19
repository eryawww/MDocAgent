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

# Predict
**scripts/predict.py**
- `from mydatasets.base_dataset import BaseDataset`
- `from agents.mdoc_agent import MDocAgent`
- Set CUDA environment variables from config
- For each agent in `mdoc_agent.agents`:
  - Load agent config from `config/agent/{agent_name}`
  - Load model config from `config/model/{model_name}`
  - Update agent_config with loaded configurations
- Load summary agent configuration:
  - Load agent config from `config/agent/{sum_agent.agent}`
  - Load model config from `config/model/{sum_agent.model}`
- Initialize BaseDataset with dataset config
- Initialize MDocAgent with mdoc_agent config
- Run prediction on dataset using `mdoc_agent.predict_dataset`:
  - Load samples with retrieval results
  - For each sample in dataset:
    - Load sample's text and image data with retrieval results
    - Run prediction through multi-agent system:
      1. General Agent: Analyzes question and provides initial response
      2. Critical Agent: Reflects on general response to identify key text/image aspects
      3. Text Agent: Focuses on text-specific aspects using critical reflection
      4. Image Agent: Focuses on image-specific aspects using critical reflection
      5. Summary Agent: Combines all agent responses into final answer
    - Save prediction results periodically (every `save_freq` samples)
    - Clean up GPU memory after each prediction
  - Save final results to dataset output path

# Dataset
data/FetaTab/samples.json   1016
data/LongDocURL/samples.json   2325
data/MMLongBench/samples.json   1073
data/PaperTab/samples.json   393
data/PaperText/samples.json   2804