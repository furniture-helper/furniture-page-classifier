import config
from custom_types.PageClassificationResult import PageClassificationResult
from models.Page import Page
from services.PostgresConnector import PostgresConnector


def get_unclassified_pages(postgres: PostgresConnector) -> list[Page]:
    query = """
        SELECT minimized_pages.url, minimized_pages.s3_key
        FROM minimized_pages
            LEFT JOIN page_classifications ON minimized_pages.url = page_classifications.url
            LEFT JOIN pages ON minimized_pages.url = pages.url
        WHERE (page_classifications.last_classified_at < minimized_pages.last_minimized_at AND page_classifications.last_classified_at < pages.last_crawled_at)
           OR page_classifications.last_classified_at IS NULL
        ORDER BY minimized_pages.last_minimized_at ASC, minimized_pages.url ASC, minimized_pages.s3_key ASC
        LIMIT %s
    """

    result = postgres.run_query(query, (config.get_page_retrieval_count(),))

    pages: list[Page] = []
    for row in result:
        pages.append(Page(row[0], row[1]))

    return pages

def update_page_classifications_batch(postgres_connector: PostgresConnector, results: list[PageClassificationResult]):
    query = """
        INSERT INTO page_classifications (url, s3_key, type, last_classified_at, created_at, updated_at)
        VALUES %s
        ON CONFLICT (url)
        DO UPDATE SET
            s3_key = EXCLUDED.s3_key,
            type = EXCLUDED.type,
            last_classified_at = EXCLUDED.last_classified_at,
            updated_at = EXCLUDED.updated_at
    """
    params_seq = [(r.url, r.s3_key, r.classification) for r in results]
    postgres_connector.execute_many(query, params_seq, template="(%s, %s, %s, NOW(), NOW(), NOW())")

