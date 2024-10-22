# KGR3
Code Repository of our paper - Retrival, Reasoning, Re-ranking: A Context-Enriched Framework for Knowledge Graph Completion

Step 1. Generate inference results with base KGC models. (Please refer to their official implementations)

Step 2. Perform Reasoning
python3 reasoning.py --rank --model_name Llama3 --model_path /path/to/LLM --dataset FB15k237 --fs_ctx --fsl 2 --start_idx 0 --end_idx 20466

python3 reasoning.py --rank --model_name Llama3 --model_path /path/to/LLM --dataset WN18RR --fs_ctx --fsl 2 --start_idx 0 --end_idx 3134

Step 3. Construct training prompts
To accelerate the prompt construction process, you may change the values of parameters '--start' and '--end', and combines the jsonl files together. 
python3 rrr_kgc_mc_train_prompt_construction.py --dataset FB15k237 --reasoning --use_ctx --num_nb_tps 10 --start 0 --end 272115 --rerank_scope 10

python3 rrr_kgc_mc_train_prompt_construction.py --dataset WN18RR --reasoning --use_ctx --num_nb_tps 10 --start 0 --end 86835 --rerank_scope 10

Step 4. Conduct SFT with LLaMA Factory
Please see the official github page of LLaMA Factory for reference. 

Step 5. Construct test (evaluation) promptsTo accelerate the prompt construction process, you may change the values of parameters '--start' and '--end', and combines the jsonl files together. 
python3 rrr_kgc_mc_prompt_construction.py --dataset FB15k237 --reasoning --use_ctx --use_reasoning_ctx --kge NBFnet --llm Llama3 --num_nb_tps 10 --start 0 --end 20466 --rerank_scope 10

python3 rrr_kgc_mc_prompt_construction.py --dataset WN18RR --reasoning --use_ctx --use_reasoning_ctx --kge SimKGC --llm Llama3 --num_nb_tps 10 --start 0 --end 3134 --rerank_scope 10

Step 6. Evaluation
python3 rrr_kgc_mc_test.py --dataset FB15k237 --use_ctx --reasoning --use_reasoning_ctx --kge NBFnet --llm Llama3 --start 0 --end 20466 --llm_batch_size 2 --model_path /path/to/LLM

python3 rrr_kgc_mc_test.py --dataset WN18RR --use_ctx --reasoning --use_reasoning_ctx --kge SimKGC --llm Llama3 --start 0 --end 3134 --llm_batch_size 2 --model_path /path/to/LLM
