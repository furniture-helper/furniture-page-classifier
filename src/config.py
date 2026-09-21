import os

DEFAULT_PAGE_RETRIEVAL = 100

def get_page_retrieval_count() -> int:
    count = os.environ.get("PAGE_RETRIEVAL_COUNT")
    if count is not None:
        try:
            return int(count)
        except ValueError:
            raise ValueError(f"Invalid PAGE_RETRIEVAL_COUNT value: {count}. Must be an integer.")
    return DEFAULT_PAGE_RETRIEVAL


def get_minimized_pages_bucket_name() -> str:
    minimized_pages_bucket_name = os.environ.get("MINIMIZED_PAGES_BUCKET_NAME")
    if minimized_pages_bucket_name is None:
        raise ValueError("MINIMIZED_PAGES_BUCKET_NAME environment variable is not set.")
    return minimized_pages_bucket_name

def get_processor_model_id() -> str:
    processor_model_id = os.environ.get("PROCESSOR_MODEL_ID")
    if processor_model_id is None:
        raise ValueError("PROCESSOR_MODEL_ID environment variable is not set.")
    return processor_model_id

def get_models_dir() -> str:
    return "../.models"

def get_classification_model_id() -> str:
    classification_model_id = os.environ.get("CLASSIFICATION_MODEL_ID")
    if classification_model_id is None:
        raise ValueError("CLASSIFICATION_MODEL_ID environment variable is not set.")
    return classification_model_id
