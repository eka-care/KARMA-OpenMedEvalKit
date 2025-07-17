---
title: Caching
---
KARMA saves the model's predictions locally to avoid redundant computations.
This ensures that running multiple metrics or extending datasets is trivial.

## How are items cached?
KARMA caches at a sample level for each evaluated model + configuration and dataset combinations.
For example, if we run evalution on pubmedqa with the Qwen3-0.6B model, we will cache for each of the configurations.
So if temperature is changed and evalution is run once again, then model will be invoked again.
However, if only a new metric has been added along with exact_match on the dataset, then the cached model outputs are reused.

Caching is hugely beneficial for ASR related models as well since the metric computation also evolves over time.
For example, if we run evaluation on a dataset with a new metric, the cached model outputs are reused.


## DuckDB Caching
DuckDB is a lightweight, in-memory, columnar database that is used by KARMA to cache the model's predictions.
This the default way of caching.

## DynamoDB Caching
DynamoDB is a NoSQL database service provided by Amazon Web Services (AWS).
KARMA can also use DynamoDB to cache model predictions.
This is useful for large-scale deployments where the model predictions need to be stored in a highly scalable and durable manner.

To use DynamoDB caching, you need to configure the following environment variables:

- `AWS_ACCESS_KEY_ID`: Your AWS access key ID.
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key.
- `AWS_REGION`: The AWS region where your DynamoDB table is located.

Once you have configured these environment variables, you can enable DynamoDB caching by setting the `KARMA_CACHE_TYPE` environment variable to `dynamodb`.
