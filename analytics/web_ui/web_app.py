import os
import sys

# Add mtmc_analytics to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
mtmc_analytics_dir = os.path.abspath(os.path.join(current_dir, '../mtmc_analytics'))
if mtmc_analytics_dir not in sys.path:
    sys.path.append(mtmc_analytics_dir)

from flask import Flask, render_template, request, jsonify, send_from_directory
from mtmc_reid.configs.app_config import load_config
from mtmc_reid.roi_analyzer.roi_analyzer_milvus import ROIAnalyzer, run_dual_mode_analysis
from mtmc_reid.roi_analyzer.roi_config import ROIConfig
from exit_ranking_milvus import run_exit_ranking

OUTPUT_DIR = os.path.join(current_dir, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(OUTPUT_DIR)

app = Flask(__name__, template_folder=os.path.join(current_dir, 'templates'))
app_config = load_config(os.path.join(mtmc_analytics_dir, "app_config.yaml"))
app_config.ROI_CONFIG_PATH = os.path.join(mtmc_analytics_dir, "roi_config.yaml")

# Configure Nginx Image Server URL (assumes it's running on host/docker port 8080)
IMAGE_SERVER_URL = os.environ.get("IMAGE_SERVER_URL", "http://localhost:8080")

@app.route('/')
def index():
    return render_template('index.html', image_server_url=IMAGE_SERVER_URL)

@app.route('/run_roi', methods=['POST'])
def run_roi():
    try:
        roi_config_path = app_config.ROI_CONFIG_PATH
        roi_config = ROIConfig.load_from_file(app_config)
        
        # This will write output files to the configured output paths relative to where it's run
        run_dual_mode_analysis(roi_config, app_config)
        
        # Ensure we point to the combined visualization file
        # The script saves it typically as 'combined_roi_analysis.jpg' in current dir or output dir
        # By default in script it's saved to combined_roi_analysis.jpg 
        output_image = "combined_roi_analysis.jpg"
        
        return jsonify({
            "status": "success", 
            "message": "ROI Analysis completed successfully.",
            "image_url": f"/results/{output_image}",
            "images": [
                {"name": "Combined Analysis", "url": f"/results/{output_image}"},
                {"name": "Enter ROI Analysis", "url": "/results/roi_transition_enter_annotated.jpg"},
                {"name": "Exit ROI Analysis", "url": "/results/roi_transition_exit_annotated.jpg"}
            ]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/run_exit_ranking', methods=['POST'])
def run_exit_ranking_api():
    try:
        data = request.get_json() or {}
        threshold = data.get('threshold', 0.60)
        
        result = run_exit_ranking(threshold=threshold, app_config=app_config)
        
        # Modify the image paths in the results to point to the nginx server
        # The paths saved in milvus look like /opt/storage/output/app/images/...
        # Our Nginx server maps /opt/storage/output to its web root /usr/share/nginx/html/opt/storage/output
        # Wait, if nginx serves from /usr/share/nginx/html/opt/storage/output/ and maps volume to /opt/storage/output
        # The url on the browser would literally be http://localhost:8080/opt/storage/output/...
        
        if result["status"] == "success":
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve files generated in the current execution directory"""
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    # Run the app binding to all interfaces for Docker access
    app.run(host='0.0.0.0', port=5000, debug=False)
