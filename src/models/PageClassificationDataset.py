from custom_types.PageClassificationResult import PageClassificationResult
from models.Page import Page
from services.ClassificationModel import ClassificationModel
from services.Logging import LoggingService
from services.Processor import Processor
from services.S3Bucket import S3Bucket
from threading import Lock


class PageClassificationDataset:

    logger = LoggingService.get_logger("PageClassificationDataset")

    def __init__(self, s3_bucket: S3Bucket, processor: Processor, model: ClassificationModel):
        self.s3_bucket = s3_bucket
        self.processor = processor
        self.model = model
        self.dataset: list[dict] = []
        self._lock = Lock()

    def add(self, page: Page):
        page_content = self.s3_bucket.download(page.s3_key)
        tokens = self.processor.tokenize(page_content)
        with self._lock:
            self.dataset.append({
                "page": page,
                "tokens": tokens
            })
        self.logger.info(f"Page {page.url} added")

    def run_inference(self):
        self.logger.info(f"Running Inference on {len(self.dataset)} pages")
        count = 0
        for datum in self.dataset:
            classification, confidence = self.model.classify(datum["tokens"])
            datum["classification"] = classification
            datum["confidence"] = confidence
            datum.pop("tokens")
            count += 1
            self.logger.info(f"[{count}/{len(self.dataset)}] Page {datum['page'].url} classified as {classification} with confidence {confidence}")

    def get_results(self) -> list[PageClassificationResult]:
        results = []
        for datum in self.dataset:
            res = PageClassificationResult(
                url=datum["page"].url,
                s3_key=datum["page"].s3_key,
                classification=datum["classification"],
            )
            results.append(res)
        return results

