import cv2
import pandas as pd
import sqlite3
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json
import argparse
from mtmc_reid.roi_analyzer.roi_data import BoundingBox, DirectionResults, ROIConfig, TrackletTransition

TABLE_NAME = 'roi_tracklet'


class ROIAnalyzer:
    """Analyzes tracklet transitions between regions of interest for multiple modes."""
    
    def __init__(self):
        """Initialize the ROI analyzer."""
        self.results: Dict[str, DirectionResults] = {}
    
    @staticmethod
    def load_tracklet_data(db_path: str, direction: str = None) -> pd.DataFrame:
        """
        Load tracklet data from SQLite database.
        
        Args:
            db_path: Path to SQLite database
            direction: Direction filter for tracklets (None for all)
            
        Returns:
            DataFrame with tracklet data sorted by id and frame
        """
        conn = sqlite3.connect(db_path)
        
        if direction:
            query = f"""
            SELECT 
                trackletid AS id,
                frameid AS frame,
                left AS x,
                top AS y,
                width AS w,
                height AS h,
                visibility,
                direction
            FROM {TABLE_NAME} 
            WHERE direction = ?
            """
            df = pd.read_sql_query(query, conn, params=(direction,))
        else:
            query = f"""
            SELECT 
                trackletid AS id,
                frameid AS frame,
                left AS x,
                top AS y,
                width AS w,
                height AS h,
                visibility,
                direction
            FROM {TABLE_NAME}
            """
            df = pd.read_sql_query(query, conn)
        
        df.sort_values(by=['id', 'frame'], inplace=True)
        conn.close()
        
        return df
    
    @staticmethod
    def point_in_polygon(point: Tuple[int, int], polygon: np.ndarray) -> bool:
        """Check if a point is inside a polygon."""
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def analyze_all_modes(self, config: ROIConfig) -> Dict[str, DirectionResults]:
        """
        Analyze tracklet transitions for all specified modes.
        
        Args:
            config: ROI configuration with mode-specific settings
            
        Returns:
            Dictionary mapping mode to ModeResults
        """
        # Load all data at once for efficiency
        all_data = self.load_tracklet_data(config.db_path)
        
        for mode in config.modes:
            print(f"Analyzing mode: {mode}")
            mode_data = all_data[all_data['direction'] == mode].copy()
            
            if mode_data.empty:
                print(f"Warning: No data found for mode '{mode}'")
                self.results[mode] = DirectionResults(
                    direction=mode,
                    transitions=[],
                    non_transit_ids=[],
                    transit_ids=[],
                    total_tracklets=0
                )
                continue
            
            # Get mode-specific configuration
            mode_config = config.get_mode_config(mode)
            roi1 = np.array(mode_config.roi1, dtype=np.int32)
            roi2 = np.array(mode_config.roi2, dtype=np.int32)
            
            transitions = []
            non_transit_ids = []
            transit_ids = []
            
            for track_id in mode_data['id'].unique():
                tracklet = mode_data[mode_data['id'] == track_id].reset_index(drop=True)
                
                # Check if tracklet starts in ROI1
                first_row = tracklet.iloc[0]
                first_bbox = BoundingBox(
                    int(first_row['x']), int(first_row['y']),
                    int(first_row['w']), int(first_row['h'])
                )
                
                if not self.point_in_polygon(first_bbox.center, roi1):
                    non_transit_ids.append(track_id)
                    continue  # Skip tracklets that don't start in ROI1
                
                transition = self._process_tracklet(tracklet, track_id, first_bbox, mode, roi1, roi2)
                
                if transition:
                    transitions.append(transition)
                    transit_ids.append(track_id)
                else:
                    non_transit_ids.append(track_id)
            
            self.results[mode] = DirectionResults(
                direction=mode,
                transitions=transitions,
                non_transit_ids=non_transit_ids,
                transit_ids=transit_ids,
                total_tracklets=len(mode_data['id'].unique())
            )
        
        return self.results
    
    def _process_tracklet(self, tracklet: pd.DataFrame, track_id: int, 
                         start_bbox: BoundingBox, mode: str, roi1: np.ndarray, roi2: np.ndarray) -> Optional[TrackletTransition]:
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
                int(row['x']), int(row['y']),
                int(row['w']), int(row['h'])
            )
            
            if self.point_in_polygon(bbox.center, roi1):
                in_roi1_flag = True
                last_roi1_frame = row['frame']
            elif in_roi1_flag and self.point_in_polygon(bbox.center, roi2):
                # Transition detected: exited ROI1 and entered ROI2
                return TrackletTransition(
                    tracklet_id=track_id,
                    represent_frame=last_roi1_frame,
                    start_bbox=start_bbox,
                    end_bbox=bbox,
                    direction=mode
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
        
        mode_config = config.get_mode_config(mode)
        
        if output_path is None:
            output_path = f"roi_transition_{mode.lower()}_annotated.jpg"
        
        image = cv2.imread(mode_config.background_image)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {mode_config.background_image}")
        
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
        
        mode_config = config.get_mode_config(mode)
        
        if output_path is None:
            output_path = f"non_transit_{mode.lower()}_tracklets.jpg"
        
        # Load mode data to get tracklet details
        mode_data = self.load_tracklet_data(config.db_path, mode)
        
        image = cv2.imread(mode_config.background_image)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {mode_config.background_image}")
        
        # Draw ROIs
        roi1 = np.array(mode_config.roi1, dtype=np.int32)
        roi2 = np.array(mode_config.roi2, dtype=np.int32)
        self._draw_rois(image, roi1, roi2)
        
        results = self.results[mode]
        
        # Draw non-transit tracklets
        for track_id in results.non_transit_ids:
            tracklet = mode_data[mode_data['id'] == track_id].reset_index(drop=True)
            if not tracklet.empty:
                start_row = tracklet.iloc[0]
                
                # Start box (blue)
                x, y, w, h = map(int, [start_row['x'], start_row['y'], 
                                      start_row['w'], start_row['h']])
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
        first_mode = config.modes[0]
        first_mode_config = config.get_mode_config(first_mode)
        
        image = cv2.imread(first_mode_config.background_image)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {first_mode_config.background_image}")
        
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
            mode_config = config.get_mode_config(mode)
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


def create_sample_config():
    """Create sample configuration files for dual-mode analysis."""
    
    # Sample JSON config for dual mode with separate ROIs
    json_config = {
        "base_path": "/home/bb24902/runOutput/20MarOutput/",
        "db_path": "/home/bb24902/runOutput/20MarOutput/mydata.db",
        "modes": ["Enter", "Exit"],
        "enter_config": {
            "roi1": [
                [120, 50],
                [1200, 50],
                [1200, 700],
                [800, 700],
                [800, 350],
                [120, 350]
            ],
            "roi2": [
                [1, 350],
                [750, 350],
                [750, 750],
                [1250, 750],
                [1250, 350],
                [1550, 350],
                [1550, 1060],
                [1, 1060]
            ],
            "background_image": "/home/bb24902/runOutput/20MarOutput/backgroundEnter.png",
            "output_path": "/home/bb24902/runOutput/20MarOutput/roi_transitions_enter.txt"
        },
        "exit_config": {
            "roi1": [
                [500, 140],
                [1200, 140],
                [1200, 690],
                [500, 690]
            ],
            "roi2": [
                [1, 450],
                [520, 450],
                [520, 800],
                [1300, 800],
                [1300, 260],
                [1919, 260],
                [1919, 1079],
                [1, 1079]
            ],
            "background_image": "/home/bb24902/runOutput/20MarOutput/backgroundExit.png",
            "output_path": "/home/bb24902/runOutput/20MarOutput/roi_transitions_exit.txt"
        }
    }
    
    # Save sample JSON config
    with open("sample_config_dual.json", "w") as f:
        json.dump(json_config, f, indent=2)
    
    # Sample YAML config for dual mode with separate ROIs
    yaml_config = """
base_path: "/home/bb24902/runOutput/20MarOutput/"
db_path: "/home/bb24902/runOutput/20MarOutput/mydata.db"
modes: ["Enter", "Exit"]

enter_config:
  roi1:
    - [120, 50]
    - [1200, 50]
    - [1200, 700]
    - [800, 700]
    - [800, 350]
    - [120, 350]
  roi2:
    - [1, 350]
    - [750, 350]
    - [750, 750]
    - [1250, 750]
    - [1250, 350]
    - [1550, 350]
    - [1550, 1060]
    - [1, 1060]
  background_image: "/home/bb24902/runOutput/20MarOutput/backgroundEnter.png"
  output_path: "/home/bb24902/runOutput/20MarOutput/roi_transitions_enter.txt"

exit_config:
  roi1:
    - [500, 140]
    - [1200, 140]
    - [1200, 690]
    - [500, 690]
  roi2:
    - [1, 450]
    - [520, 450]
    - [520, 800]
    - [1300, 800]
    - [1300, 260]
    - [1919, 260]
    - [1919, 1079]
    - [1, 1079]
  background_image: "/home/bb24902/runOutput/20MarOutput/backgroundExit.png"
  output_path: "/home/bb24902/runOutput/20MarOutput/roi_transitions_exit.txt"
"""
    
    with open("sample_config_dual.yaml", "w") as f:
        f.write(yaml_config)
    
    print("Sample dual-mode configuration files created:")
    print("- sample_config_dual.json")
    print("- sample_config_dual.yaml")


def run_dual_mode_analysis(config: ROIConfig, 
                          custom_viz_paths: Dict[str, str] = None) -> None:
    """
    Run ROI analysis for multiple modes using configuration.
    
    Args:
        config: ROI configuration
        custom_viz_paths: Optional custom visualization paths
    """
    try:
        # Initialize analyzer
        analyzer = ROIAnalyzer()
        
        # Load and analyze data for all modes
        print(f"Loading tracklet data from: {config.db_path}")
        print(f"Analyzing modes: {config.modes}")
        
        results = analyzer.analyze_all_modes(config)
        
        # Generate visualizations for each mode
        for mode in config.modes:
            # Transition visualization
            transition_viz_path = None
            if custom_viz_paths and f"{mode}_transition" in custom_viz_paths:
                transition_viz_path = custom_viz_paths[f"{mode}_transition"]
            
            analyzer.visualize_transitions(config, mode, transition_viz_path)
            
            # Non-transit visualization
            non_transit_viz_path = None
            if custom_viz_paths and f"{mode}_non_transit" in custom_viz_paths:
                non_transit_viz_path = custom_viz_paths[f"{mode}_non_transit"]
            
            analyzer.visualize_non_transits(config, mode, non_transit_viz_path)
        
        # Generate combined visualization
        analyzer.visualize_combined_results(config)
        
        # Save results
        analyzer.save_transition_results(config)
        analyzer.save_combined_results()
        
        # Print comprehensive summary
        analyzer.print_summary()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


def main():
    """Main function with CLI support for dual-mode analysis."""
    parser = argparse.ArgumentParser(
        description="ROI Tracklet Transition Analyzer - Dual Mode with Separate ROI Configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample configuration files
  python roi_analyzer.py --create-sample-config
  
  # Run with dual-mode configuration file
  python roi_analyzer.py --config config_dual.json
  
  # Run with specific modes only
  python roi_analyzer.py --config config.json --modes Enter Exit
        """
    )
    
    parser.add_argument("--config", "-c", type=str,
                       help="Path to configuration file (JSON or YAML)")
    
    parser.add_argument("--create-sample-config", action="store_true",
                       help="Create sample configuration files and exit")
    
    parser.add_argument("--modes", nargs='+', choices=["Enter", "Exit"],
                       help="Specify modes to analyze (overrides config)")
    
    args = parser.parse_args()
    
    # Handle create-sample-config first
    if args.create_sample_config:
        create_sample_config()
        return
    
    # Now check for required config argument
    if not args.config:
        parser.error("--config/-c is required when not using --create-sample-config")
    
    # Load configuration from file
    print(f"Loading configuration from: {args.config}")
    config = ROIConfig.load_from_file(args.config)
    
    # Override modes if specified
    if args.modes:
        config.modes = args.modes
        print(f"Modes overridden to: {args.modes}")
    
    run_dual_mode_analysis(config)


if __name__ == "__main__":
    main()