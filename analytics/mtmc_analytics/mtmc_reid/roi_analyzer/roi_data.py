from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class BoundingBox:
    """Represents a bounding box with position and dimensions."""
    x: int
    y: int
    w: int
    h: int
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return the center point of the bounding box."""
        return (self.x + self.w // 2, self.y + self.h // 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (x, y, w, h) tuple."""
        return (self.x, self.y, self.w, self.h)


@dataclass
class TrackletTransition:
    """Represents a tracklet transition between ROIs."""
    tracklet_id: int
    represent_frame: int
    start_bbox: BoundingBox
    end_bbox: BoundingBox
    direction: str  # "Enter" or "Exit"


@dataclass
class DirectionResults:
    """Results for a specific direction (Enter/Exit)."""
    direction: str
    transitions: List[TrackletTransition]
    non_transit_ids: List[int]
    
    total_tracklets: int
    
    @property
    def transition_count(self) -> int:
        return len(self.transitions)
    
    @property
    def non_transit_count(self) -> int:
        return len(self.non_transit_ids)

    @property
    def transit_ids(self) -> List[int]:
        return [transition.tracklet_id for transition in self.transitions]
    @property
    def dict_transitid_representframeid(self) -> dict[int:int]:
        dict_id_frame = {}
        for transition in self.transitions:
            dict_id_frame[transition.tracklet_id] = transition.represent_frame
        return dict_id_frame




