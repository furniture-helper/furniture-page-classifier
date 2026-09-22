from dataclasses import dataclass


@dataclass
class PageClassificationResult:
    url: str
    s3_key: str
    classification: str
    confidence: float

    def __str__(self):
        return f"(url={self.url}, classification={self.classification})"