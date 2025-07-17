---
title: Core Components of KARMA
---
This document defines the four core components of KARMA's evaluation system and how they interact with each other.
1. Models
2. Datasets
3. Metrics
4. Processors

## Data Flow Sequence

```mermaid
sequenceDiagram
    participant CLI
    participant Orchestrator
    participant Registry
    participant Model
    participant Dataset
    participant Processor
    participant Metrics
    participant Cache

    CLI->>Orchestrator: karma eval model --datasets ds1
    Orchestrator->>Registry: discover_all_registries()
    Registry-->>Orchestrator: components metadata

    Orchestrator->>Model: initialize with config
    Orchestrator->>Dataset: initialize with args
    Orchestrator->>Processor: initialize processors

    loop For each dataset
        Orchestrator->>Dataset: create dataset instance
        Dataset->>Processor: apply postprocessors

        loop For each batch
            Dataset->>Model: provide samples
            Model->>Cache: check cache

            alt Cache miss
                Model->>Model: run inference
                Model->>Cache: store results
            end

            Model-->>Dataset: return predictions
            Dataset->>Dataset: extract_prediction()
            Dataset->>Processor: postprocess predictions
            Processor-->>Dataset: processed text

            Dataset->>Metrics: evaluate(predictions, references)
            Metrics-->>Dataset: scores
        end

        Dataset-->>Orchestrator: evaluation results
    end

    Orchestrator-->>CLI: aggregated results
```

## Component Interaction Diagram

```mermaid
graph TD
    %% CLI Layer
    CLI[CLI Command<br/>karma eval model --datasets ds1,ds2]

    %% Orchestrator Layer
    ORCH[Orchestrator<br/>MultiDatasetOrchestrator]

    %% Registry System
    MR[Model Registry]
    DR[Dataset Registry]
    MetR[Metrics Registry]
    PR[Processor Registry]

    %% Core Components
    MODEL[Model<br/>BaseModel]
    DATASET[Dataset<br/>BaseMultimodalDataset]
    METRICS[Metrics<br/>BaseMetric]
    PROC[Processors<br/>BaseProcessor]

    %% Benchmark
    BENCH[Benchmark<br/>Evaluation Engine]

    %% Cache System
    CACHE[Cache Manager<br/>DuckDB/DynamoDB]

    %% Data Flow
    CLI --> |parse args| ORCH

    ORCH --> |discover| MR
    ORCH --> |discover| DR
    ORCH --> |discover| MetR
    ORCH --> |discover| PR

    MR --> |create| MODEL
    DR --> |create| DATASET
    MetR --> |create| METRICS
    PR --> |create| PROC

    ORCH --> |orchestrate| BENCH

    BENCH --> |inference| MODEL
    BENCH --> |iterate| DATASET
    BENCH --> |compute| METRICS
    BENCH --> |cache lookup/store| CACHE

    DATASET --> |postprocess| PROC
    DATASET --> |extract predictions| MODEL

    MODEL --> |predictions| DATASET
    DATASET --> |processed data| METRICS
    PROC --> |normalized text| METRICS

    %% Configuration Flow
    CLI -.-> |--model-args| MODEL
    CLI -.-> |--dataset-args| DATASET
    CLI -.-> |--metric-args| METRICS
    CLI -.-> |--processor-args| PROC

    %% Styling
    classDef cli fill:#e1f5fe
    classDef orchestrator fill:#f3e5f5
    classDef registry fill:#fff3e0
    classDef component fill:#e8f5e8
    classDef benchmark fill:#fff8e1
    classDef cache fill:#fce4ec

    class CLI cli
    class ORCH orchestrator
    class MR,DR,MetR,PR registry
    class MODEL,DATASET,METRICS,PROC component
    class BENCH benchmark
    class CACHE cache
```


This architecture ensures clean separation of concerns while enabling flexible configuration and robust error handling throughout the evaluation process.
