import cv2
import pandas as pd
from pymilvus import MilvusClient
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json
import argparse
from mtmc_reid.roi_analyzer.roi_data import BoundingBox, DirectionResults, TrackletTransition
from mtmc_reid.roi_analyzer.roi_config import ROIConfig
from mtmc_reid.configs.app_config import AppConfig, load_config
from mtmc_reid.database.tracklet_schema import TrackletRecord
class ROIAnalyzer:
    """Analyzes tracklet transitions between regions of interest for multiple modes."""
    
    def __init__(self):
        """Initialize the ROI analyzer."""
        self.results: Dict[str, DirectionResults] = {}
    
    @staticmethod
    def load_tracklet_data(config: ROIConfig, direction: str = None) -> pd.DataFrame:
        """
        Load tracklet data from Milvus database.
        
        Args:
            config: ROIConfig 
            direction: Direction filter for tracklets (None for all)
            
        Returns:
            DataFrame with tracklet data sorted by id and frameid
        """
        
        client = MilvusClient(config.db_path)

        if direction:
            expr = f'direction == "{direction}"'

        else:
            expr = 'direction == "Enter" or direction == "Exit"'
        iterator = client.query_iterator(
                    collection_name=config.collection_name,
                    filter=expr,
                    output_fields=['trackingId', 'frameid', 'direction',
                                    'bbox_top', 'bbox_left', 'bbox_width', 'bbox_height']
                    )
        results = []
        while True:
            result = iterator.next()
            if not result:
                iterator.close()
                break
            results += result
        client.close()
        df = pd.DataFrame.from_records(results)
        df.sort_values(by=['trackingId', 'frameid'], inplace=True)
        return df
    
    @staticmethod
    def point_in_polygon(point: Tuple[int, int], polygon: np.ndarray) -> bool:
        """Check if a point is inside a polygon."""
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    @staticmethod
    def chunked(iterable, batch_size):
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i + batch_size]
    def analyze_all_modes(self, config: ROIConfig) -> Dict[str, DirectionResults]:
        """
        Analyze tracklet transitions for all specified modes.
        
        Args:
            config: ROI configuration with mode-specific settings
            
        Returns:
            Dictionary mapping mode to ModeResults
        """
        # Load all data at once for efficiency
        all_data = self.load_tracklet_data(config)
        
        for direction in config.directions:
            print(f"Analyzing direction: {direction}")
            direction_data = all_data[all_data['direction'] == direction].copy()
            
            if direction_data.empty:
                print(f"Warning: No data found for mode '{direction}'")
                self.results[direction] = DirectionResults(
                    direction=direction,
                    transitions=[],
                    non_transitions=[],
                    total_tracklets=0
                )
                continue
            
            # Get mode-specific configuration
            direction_config = config.get_direction_config(direction)
            roi1 = np.array(direction_config.roi1, dtype=np.int32)
            roi2 = np.array(direction_config.roi2, dtype=np.int32)
            
            transitions = []
            non_transit_ids = []
            for track_id in direction_data['trackingId'].unique():
                tracklet = direction_data[direction_data['trackingId'] == track_id].reset_index(drop=True)
                
                # Check if tracklet starts in ROI1
                first_row = tracklet.iloc[0]
                first_bbox = BoundingBox(
                    int(first_row['bbox_left']), int(first_row['bbox_top']),
                    int(first_row['bbox_width']), int(first_row['bbox_height'])
                )
                
                if not self.point_in_polygon(first_bbox.center, roi1):
                    non_transit_ids.append(track_id)
                    continue  # Skip tracklets that don't start in ROI1
                
                transition = self._process_tracklet(tracklet, track_id, first_bbox, direction, roi1, roi2)
                
                if transition:
                    transitions.append(transition)
                else:
                    non_transit_ids.append(track_id)
            
            self.results[direction] = DirectionResults(
                direction=direction,
                transitions=transitions,
                non_transit_ids=non_transit_ids,
                total_tracklets=len(direction_data['trackingId'].unique())
            )
        
        return self.results
    
    def _process_tracklet(self, tracklet: pd.DataFrame, track_id: int, 
                         start_bbox: BoundingBox, direction_param: str, roi1: np.ndarray, roi2: np.ndarray) -> Optional[TrackletTransition]:
        """
        Process a single tracklet to detect ROI1 -> ROI2 transition.
        
        Args:
            tracklet: DataFrame for single tracklet
            track_id: Tracklet ID
            start_bbox: Starting bounding box
            mode: Analysis mode
            roi1: ROI1 polygon for this mode
            roi2: ROI2 polygon for this mode
            
        Returns:
            TrackletTransition if transition found, None otherwise
        """
        in_roi1_flag = False
        last_roi1_frame = None
        
        for _, row in tracklet.iterrows():
            bbox = BoundingBox(
                int(row['bbox_left']), int(row['bbox_top']),
                int(row['bbox_width']), int(row['bbox_height'])
            )
            
            if self.point_in_polygon(bbox.center, roi1):
                in_roi1_flag = True
                last_roi1_frame = row['frameid']
            elif in_roi1_flag and self.point_in_polygon(bbox.center, roi2):
                # Transition detected: exited ROI1 and entered ROI2
                return TrackletTransition(
                    tracklet_id=track_id,
                    represent_frame=tracklet.iloc[0]['frameid'],
                    start_bbox=start_bbox,
                    end_bbox=bbox,
                    direction=direction_param
                )
        
        return None
    
    def visualize_transitions(self, config: ROIConfig, mode: str, 
                            output_path: str = None) -> None:
        """
        Visualize ROI transitions for a specific mode.
        
        Args:
            config: ROI configuration
            mode: Mode to visualize
            output_path: Path to save the visualization
        """
        if mode not in self.results:
            print(f"No results found for mode: {mode}")
            return
        
        mode_config = config.get_direction_config(mode)
        
        if output_path is None:
            output_path = f"roi_transition_{mode.lower()}_annotated.jpg"
        
        image = cv2.imread(mode_config.background_image)
        if image is None:
            print(f"Warning: Could not load image {mode_config.background_image}. Using blank canvas.")
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Draw ROIs
        roi1 = np.array(mode_config.roi1, dtype=np.int32)
        roi2 = np.array(mode_config.roi2, dtype=np.int32)
        self._draw_rois(image, roi1, roi2)
        
        results = self.results[mode]
        
        # Draw transitions
        for transition in results.transitions:
            # End box (orange) - final position
            start_bbox = transition.start_bbox
            cv2.putText(image, f"ID:{transition.tracklet_id}", 
                       (start_bbox.x, start_bbox.y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(image, (start_bbox.x, start_bbox.y), 
                         (start_bbox.x + start_bbox.w, start_bbox.y + start_bbox.h), 
                         (0, 165, 255), 2)
        
        # Add mode and count information
        cv2.putText(image, f"Mode: {mode} | Transitions: {results.transition_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, image)
        print(f"Transition visualization for {mode} saved to: {output_path}")
    
    def visualize_non_transits(self, config: ROIConfig, mode: str,
                              output_path: str = None) -> None:
        """
        Visualize non-transit tracklets for a specific mode.
        
        Args:
            config: ROI configuration
            mode: Mode to visualize
            output_path: Path to save the visualization
        """
        if mode not in self.results:
            print(f"No results found for mode: {mode}")
            return
        
        mode_config = config.get_direction_config(mode)
        
        if output_path is None:
            output_path = f"non_transit_{mode.lower()}_tracklets.jpg"
        
        # Load mode data to get tracklet details
        mode_data = self.load_tracklet_data(config, mode)
        
        image = cv2.imread(mode_config.background_image)
        if image is None:
            print(f"Warning: Could not load image {mode_config.background_image}. Using blank canvas.")
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Draw ROIs
        roi1 = np.array(mode_config.roi1, dtype=np.int32)
        roi2 = np.array(mode_config.roi2, dtype=np.int32)
        self._draw_rois(image, roi1, roi2)
        
        results = self.results[mode]
        
        # Draw non-transit tracklets
        for track_id in results.non_transit_ids:
            tracklet = mode_data[mode_data['trackingId'] == track_id].reset_index(drop=True)
            if not tracklet.empty:
                start_row = tracklet.iloc[0]
                
                # Start box (blue)
                x, y, w, h = map(int, [start_row['bbox_left'], start_row['bbox_top'], 
                                      start_row['bbox_width'], start_row['bbox_height']])
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(image, f"ID:{track_id}", (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add mode and count information
        cv2.putText(image, f"Mode: {mode} | Non-Transits: {results.non_transit_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, image)
        print(f"Non-transit visualization for {mode} saved to: {output_path}")
    
    def visualize_combined_results(self, config: ROIConfig, 
                                 output_path: str = "combined_roi_analysis.jpg") -> None:
        """
        Create a combined visualization showing results for all modes.
        
        Args:
            config: ROI configuration
            output_path: Path to save the combined visualization
        """
        if not self.results:
            print("No results to visualize")
            return
        
        # Use the first available mode's background image as base
        first_mode = config.directions[0]
        first_mode_config = config.get_direction_config(first_mode)
        
        image = cv2.imread(first_mode_config.background_image)
        if image is None:
            print(f"Warning: Could not load image {first_mode_config.background_image}. Using blank canvas.")
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Draw ROIs for the first mode
        roi1 = np.array(first_mode_config.roi1, dtype=np.int32)
        roi2 = np.array(first_mode_config.roi2, dtype=np.int32)
        self._draw_rois(image, roi1, roi2)
        
        # Add summary text
        y_offset = 30
        for mode, results in self.results.items():
            text = f"{mode}: {results.transition_count} transitions, {results.non_transit_count} non-transits"
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        cv2.imwrite(output_path, image)
        print(f"Combined visualization saved to: {output_path}")
    
    def _draw_rois(self, image: np.ndarray, roi1: np.ndarray, roi2: np.ndarray) -> None:
        """Draw ROIs on the image."""
        # Draw ROI1 (polygon) in magenta
        cv2.polylines(image, [roi1.reshape((-1, 1, 2))], 
                     True, (255, 0, 255), 2)
        # Draw ROI2 (polygon) in yellow
        cv2.polylines(image, [roi2.reshape((-1, 1, 2))], 
                     True, (0, 255, 255), 2)
    
    def save_transition_results(self, config: ROIConfig) -> None:
        """
        Save transition results for all modes to separate files.
        
        Args:
            config: ROI configuration with output paths
        """
        for mode, results in self.results.items():
            mode_config = config.get_direction_config(mode)
            output_path = mode_config.output_path
            
            with open(output_path, "w") as f:
                for transition in results.transitions:
                    f.write(f"{transition.tracklet_id},{transition.represent_frame}\n")
            
            print(f"Transition results for {mode} saved to: {output_path}")
    
    def save_combined_results(self, output_path: str = "roi_transitions_combined.txt") -> None:
        """
        Save combined results for all modes to a single file.
        
        Args:
            output_path: Path to save the combined results
        """
        with open(output_path, "w") as f:
            f.write("Mode,TrackletID,LastROI1Frame\n")  # Header
            for mode, results in self.results.items():
                for transition in results.transitions:
                    f.write(f"{mode},{transition.tracklet_id},{transition.represent_frame}\n")
        
        print(f"Combined transition results saved to: {output_path}")
    
    def print_summary(self) -> None:
        """Print a comprehensive summary of all results."""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*50)
        print("ROI TRANSITION ANALYSIS SUMMARY")
        print("="*50)
        
        total_transitions = 0
        total_non_transits = 0
        total_tracklets = 0
        
        for mode, results in self.results.items():
            print(f"\n{mode} Mode:")
            print(f"  Total tracklets: {results.total_tracklets}")
            print(f"  ROI1 → ROI2 transitions: {results.transition_count}")
            print(f"  Non-transit tracklets: {results.non_transit_count}")
            print(f"  Transition rate: {results.transition_count/results.total_tracklets*100:.1f}%" 
                  if results.total_tracklets > 0 else "  Transition rate: N/A")
            
            total_transitions += results.transition_count
            total_non_transits += results.non_transit_count
            total_tracklets += results.total_tracklets
        
        print(f"\nOverall Summary:")
        print(f"  Total tracklets (all modes): {total_tracklets}")
        print(f"  Total transitions (all modes): {total_transitions}")
        print(f"  Total non-transits (all modes): {total_non_transits}")
        print("="*50)

    def reset_db_roi(self,app_config: AppConfig) -> None:
        client = MilvusClient(app_config.DEFAULT_SQL_DB)
        
        iterator = client.query_iterator(
                    collection_name = app_config.COLLECTION_NAME,
                    filter=f'isTransit == True',
                    output_fields=['*']
                    )
        results = []
        while True:
            result = iterator.next()
            if not result:
                iterator.close()
                print("Finish query for reset db")
                break
            results += result
        tr_list = results
        batch_data = []
        for tr in tr_list:
            tr['isTransit'] = False
            tr['isRepresentative'] = False
            batch_data.append(tr)

        if batch_data:
            BATCH_SIZE = 10000 
            for chunk in self.chunked(batch_data, BATCH_SIZE):
                res = client.upsert(
                    collection_name=app_config.COLLECTION_NAME,
                    data=chunk
                )
                print(res)

        client.close()
    def save_to_db(self,app_config: AppConfig, results: List[DirectionResults]) -> None:
        client = MilvusClient(app_config.DEFAULT_SQL_DB)
        print(results)
        for dirResult in results:
            print(f'{dirResult.direction} trackingId in {dirResult.transit_ids}')
            if len(dirResult.transit_ids) == 0:
                print("Warning No Transit Id")
                continue
            represent_frames = dirResult.dict_transitid_representframeid
            formatted_ids = [int(tid) for tid in dirResult.transit_ids]
            iterator = client.query_iterator(
                        collection_name = app_config.COLLECTION_NAME,
                        filter=f'trackingId in {formatted_ids}',
                        output_fields=['*']
                        )
            results = []
            while True:
                result = iterator.next()
                if not result:
                    iterator.close()
                    print("Finish query for saving db")
                    break
                results += result
            tr_list = results
            batch_data = []
            for tr in tr_list:
                tr['isTransit'] = True
                if tr['frameid'] == represent_frames.get(tr['trackingId']):
                    tr['isRepresentative'] = True
                batch_data.append(tr)

            if batch_data:
                BATCH_SIZE = 10000 
                for chunk in self.chunked(batch_data, BATCH_SIZE):
                    res = client.upsert(
                        collection_name=app_config.COLLECTION_NAME,
                        data=chunk
                    )
                    print(res)

        client.close()


def run_dual_mode_analysis(roi_config: ROIConfig, app_config: AppConfig,  
                          custom_viz_paths: Dict[str, str] = None) -> None:
    """
    Run ROI analysis for multiple modes using configuration.
    
    Args:
        roi_config: ROI configuration
        custom_viz_paths: Optional custom visualization paths
    """
    try:
        # Initialize analyzer
        analyzer = ROIAnalyzer()
        analyzer.reset_db_roi(app_config)
        # Load and analyze data for all modes
        print(f"Loading tracklet data from: {roi_config.db_path}")
        print(f"Analyzing modes: {roi_config.directions}")
        
        results = analyzer.analyze_all_modes(roi_config)
        
        # Generate visualizations for each mode
        for mode in roi_config.directions:
            # Transition visualization
            transition_viz_path = None
            if custom_viz_paths and f"{mode}_transition" in custom_viz_paths:
                transition_viz_path = custom_viz_paths[f"{mode}_transition"]
            
            analyzer.visualize_transitions(roi_config, mode, transition_viz_path)
            
            # Non-transit visualization
            non_transit_viz_path = None
            if custom_viz_paths and f"{mode}_non_transit" in custom_viz_paths:
                non_transit_viz_path = custom_viz_paths[f"{mode}_non_transit"]
            
            analyzer.visualize_non_transits(roi_config, mode, non_transit_viz_path)
        
        # Generate combined visualization
        analyzer.visualize_combined_results(roi_config)
        
        # Save results
        analyzer.save_transition_results(roi_config)
        analyzer.save_combined_results()
        
        # Print comprehensive summary
        analyzer.print_summary()
        merged_result = results.values()
        analyzer.save_to_db(app_config, merged_result)
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


def main():
    """Main function for dual-mode analysis."""
    
    # Load configuration from file
    app_config = load_config()
    roi_config_path = app_config.ROI_CONFIG_PATH
    print(f"Loading configuration from: {roi_config_path}")
    roi_config = ROIConfig.load_from_file(app_config)    
    run_dual_mode_analysis(roi_config ,app_config) 
if __name__ == "__main__":
    main()