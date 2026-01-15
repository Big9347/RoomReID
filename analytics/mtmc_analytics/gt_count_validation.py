import sqlite3 
class TrackletCountValidate():
    def __init__(self):
        # Database connections
        self.gt_db_con = None
        self.pred_db_con = None
        # Data storage
        self.gt_tracklets = []
    def load_tracklet_data(self):
        """Load tracklet data from both databases"""
        # Load ground truth tracklets 
        gt_query = """
        SELECT *
        FROM gt_match_with_timestamps   
        ORDER BY EnterFirstTimestamp;

        """
        self.gt_tracklets = self.gt_db_con.execute(gt_query).fetchall()
        
        # Load new tracklets
        new_query = """
        SELECT TrackletID, StartTime, EndTime, Category
        FROM gt_tracklet
        ORDER BY TrackletID
        """
        self.new_tracklets = self.pred_db_con.execute(new_query).fetchall()
        
        print(f"Loaded {len(self.gt_tracklets)} gt tracklets and {len(self.new_tracklets)} new tracklets")
    
    def start_comparison(self):
        """Initialize databases and start comparison"""
        try:
            # Connect to databases
            self.old_conn = sqlite3.connect(self.old_db_path)
            self.new_conn = sqlite3.connect(self.new_db_path)
            
            # Enable sqlite_vec for both connections
            self.old_conn.enable_load_extension(True)
            sqlite_vec.load(self.old_conn)
            self.old_conn.enable_load_extension(False)
            
            self.new_conn.enable_load_extension(True)
            sqlite_vec.load(self.new_conn)
            self.new_conn.enable_load_extension(False)
            
            self.old_cursor = self.old_conn.cursor()
            self.new_cursor = self.new_conn.cursor()
            
            # Load tracklet data
            self.load_tracklet_data()
            
            # Setup comparison widgets
            self.setup_comparison_widgets()
            
            # Start comparison
            self.show_next_query()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize databases: {str(e)}")