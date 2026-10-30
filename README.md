## Install

```bash
 pip install -r requirement.txt
```

## 1、Dataset Download

```bash
python rag/prepare_data.py
```

## 2、Configure LLM

Configure LLM related properties in the model_comfigs folder

---

Note: Please change the basepath to your own project working directory when running the following code

---

## 3、Database creation

> Note: The chromadb version in this project is 0.5.20

```bash
# Taking the construction of nfcorpus database as an example
python rag/vectorstore.py --eval_dataset 'nfcorpus' --eval_model_code "contriever" --score_function 'cosine'
```

## 4、Entity extraction [relies on LLM to extract entities, construction is relatively slow]

```bash
python entity_generate/generate_entity_llm_check.py --eval_dataset 'nfcorpus' --dataset_prob 1
```

## 5、Watermark unit generation

```bash
python entity_generate/generate_hash_entity.py --eval_dataset 'nfcorpus' -t scratch --entity_num 100 --edge_prob 0.05
```

### 6、Watermark text generation, injection, and verification

```bash
# generation
python src/main.py --eval_dataset 'nfcorpus' --eval_model_code "contriever" --score_function 'cosine' --doc 1 --inject 0 --verify 0 --stat 0 --mutual_times 10
# injection
python src/main.py --eval_dataset 'nfcorpus' --eval_model_code "contriever" --score_function 'cosine' --doc 0 --inject 1 --verify 0 --stat 0 --mutual_times 10
# verification
python src/main.py --eval_dataset 'nfcorpus' --eval_model_code "contriever" --score_function 'cosine' --doc 0 --inject 0 --verify 1 --stat 1 --mutual_times 10
```

