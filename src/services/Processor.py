from pathlib import Path
from threading import Lock

from transformers import MarkupLMProcessor

import config
from services.Logging import LoggingService


class Processor:

    logger = LoggingService.get_logger("Processor")

    def __init__(self, model_id: str):
        local_path = Path(config.get_models_dir()) / model_id
        if local_path.exists():
            self.logger.info(f"Loading model from {local_path}")
            self.processor = MarkupLMProcessor.from_pretrained(local_path, local_files_only=True)
        else:
            self.logger.info(f"Loading model from Hugging Face Hub: {model_id}")
            self.processor = MarkupLMProcessor.from_pretrained(model_id)
        self._lock = Lock()

    def tokenize(self, html_content: str) -> dict:
        with self._lock:
            encoding = self.processor(
                html_content,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
        return {
            "input_ids": encoding["input_ids"],
            "token_type_ids": encoding["token_type_ids"],
            "attention_mask": encoding["attention_mask"],
            "xpath_tags_seq": encoding["xpath_tags_seq"],
            "xpath_subs_seq": encoding["xpath_subs_seq"],
        }