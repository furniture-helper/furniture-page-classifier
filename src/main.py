import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from custom_types.PageClassificationResult import PageClassificationResult
from helpers.page_helpers import get_unclassified_pages, update_page_classifications_batch
from helpers.url_helpers import get_domain_from_url
from models.PageClassificationDataset import PageClassificationDataset
from services.ClassificationModel import ClassificationModel
from services.KafkaService import KafkaService, ClassificationEvent
from services.Logging import LoggingService
from services.PostgresConnector import PostgresConnector
from services.Processor import Processor
from services.S3Bucket import S3Bucket


def main():
    logger = LoggingService.get_logger("main")
    logger.info("Starting page classifier...")
    postgres_connector = PostgresConnector(
        host=os.environ['PG_HOST'],
        port=int(os.environ['PG_PORT']),
        user=os.environ['PG_USER'],
        password=os.environ['PG_PASSWORD'],
        database=os.environ['PG_DATABASE'],
    )
    kafkaService = KafkaService(os.getenv("KAFKA_BROKER_URLS"), "classification-events")
    minimized_pages_bucket = S3Bucket(bucket_name=config.get_minimized_pages_bucket_name())
    processor = Processor(config.get_processor_model_id())
    model = ClassificationModel(config.get_classification_model_id())
    dataset = PageClassificationDataset(minimized_pages_bucket, processor, model)

    pages = get_unclassified_pages(postgres_connector)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(dataset.add, page) for page in pages]
        for future in as_completed(futures):
            future.result()

    dataset.run_inference()
    results: list[PageClassificationResult] = dataset.get_results()

    update_page_classifications_batch(postgres_connector, results)
    for result in results:
        event = ClassificationEvent(
            url=result.url,
            classification=result.classification,
            confidence=result.confidence,
        )
        kafkaService.send_message(event.to_dict())

    logger.info(f"Page classification completed for {len(results)} pages.")


if __name__ == "__main__":
    main()

