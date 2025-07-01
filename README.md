# parrotlet-omni


karma eval \
--model Qwen/Qwen3-0.6B
--dataset openlifescienceai/pubmedqa \
--metric_config metric_config.json
--model_configs 


tasks
-- model 
    -- Qwen/Qwen3-0.6B
        -- temp: 0.0
        -- top_p: 0.9

-- task
    -- openlifescienceai/pubmedqa (MCQATask - Accuracy)
    -- mteb/medical-retrieval (RetrievalTask - nDCG, MRR)
    -- ai4bharat/IN22-Conv (TranslationTask - BLEU, WER)
        -- targetLanguage: hi
    -- eka/DocAssistSummary (EkaDocAssistLLMRubricTask)


--cache_details
    --use_cache: true
    --duck_db:
        --cache_path: /tmp/duckdb_bench.duckdb


pip install karma[audio]
pip install karma[text]
pip install karma[retrieval]
pip install karma[images]
pip install karma[all]
