# Modifications

- src/medrag.py <= Allow to use adaptive_k
- src/utils.py <= only change the 'def extract(self, ids):'
- Create adaptive_k_llm.py to enable adaptive_k conduction

# To get Corpus and its Index
Use Corpus: 'MedCorp' and retriever: 'RRF-4'

# Notes
- the data_curation.ipynb was used through Colab
- To test the code you need to deploy your own OpenAI API key in src/config.py
