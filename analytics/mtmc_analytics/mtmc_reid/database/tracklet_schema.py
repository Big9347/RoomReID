from pydantic import BaseModel, StringConstraints, model_validator, ValidationInfo, ValidationError
from typing import Annotated,Literal, Dict, Iterator


class TrackletRecord(BaseModel):
    version: Annotated[str, StringConstraints(max_length=5)]
    frameid: int
    timestamp: Annotated[str, StringConstraints(max_length=25)]
    sensorId: Annotated[str, StringConstraints(max_length=10)]
    trackingId: int
    direction: Literal['Enter', 'Exit']
    confidence: float
    bbox_top: float
    bbox_left: float
    bbox_width: float
    bbox_height: float
    objClassName: Annotated[str, StringConstraints(max_length=25)]
    imgPath: Annotated[str, StringConstraints(max_length=100)]
    embedding: list[float]
    isTransit: bool = False
    isRepresentative: bool = False
    def validate_selected_fields_from_model(model: BaseModel, data_dict: Dict) -> Iterator[ValidationError]:
        """
        Validates fields one by one, yielding any ValidationErrors encountered.
        This allows immediate error handling as soon as any validation fails.
        
        Args:
            model: The Pydantic model to validate against
            data_dict: Dictionary containing data to validate
            
        Yields:
            ValidationError: Any validation errors encountered during the process
        """
        model_con = model.model_construct()
        for k, v in data_dict.items():
            try:
                model.__pydantic_validator__.validate_assignment(model_con, k, v)
            except ValidationError as e:
                yield e
    @model_validator(mode="after")
    def check_embedding_dim(self, info: ValidationInfo) -> "TrackletRecord":
        config = info.context.get("config")
        if config is None:
            raise ValueError("AppConfig not found in validation context.")
        expected_dim = config.REID_FEATURE_DIM
        if len(self.embedding) != expected_dim:
            raise ValueError(f"Embedding must be of length {expected_dim}, got {len(self.embedding)}")
        return self

