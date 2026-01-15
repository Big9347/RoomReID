import yaml
from pydantic import BaseModel
from typing import Dict

class AppConfig(BaseModel):
    BASE_PATH: str
    REID_FEATURE_DIM: int
    LABEL_TO_CLASS_ID: Dict[str, int] = {
        "BG": 0,
        "Person": 1,
        "Face": 2,
        "Bag": 3
    }
    COLLECTION_NAME : str = "my_collection"
    KAFKA_TOPIC : str = 'mdx-raw'
    ROI_MODES : list[str] = ['Enter', 'Exit']
    ENTER_SENSORIDS : list[str]
    EXIT_SENSORIDS : list[str]
    ROI_CONFIG_PATH: str = "roi_config.yaml"
    @property
    def METADATA_FILE(self):
        return f"{self.BASE_PATH}/output/metadata.csv"

    @property
    def DEFAULT_SQL_DB(self):
        return "http://localhost:19530"



def load_config(path="app_config.yaml") -> AppConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
