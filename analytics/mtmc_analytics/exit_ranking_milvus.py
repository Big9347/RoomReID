import csv
import argparse
import os
from pymilvus import MilvusClient, DataType
from mtmc_reid.configs.app_config import load_config, AppConfig

#FEAT_DIM = 256

def parse_arguments():
    parser = argparse.ArgumentParser(description="Milvus vector similarity matching for tracklets")
    parser.add_argument("--milvus_uri", 
                       default="localhost:19530", 
                       help="Milvus server URI")
    parser.add_argument("--collection_name", 
                       default="merged_enter_exit", 
                       help="Collection name in Milvus")
    parser.add_argument("--threshold", type=float, default=0.60, 
                       help="Similarity threshold (default: 0.60)")
    return parser.parse_args()

def setup_milvus_client(uri):
    """Setup Milvus client connection"""
    return MilvusClient(uri)

def check_collection_if_not_exists(client, collection_name):
    """Create collection with proper schema if it doesn't exist"""
    if collection_name not in client.list_collections():
        # schema = MilvusClient.create_schema(
        #     auto_id=True,
        #     enable_dynamic_field=False
        # )
        
        # schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        # schema.add_field(field_name="trackingId", datatype=DataType.INT64)
        # schema.add_field(field_name="frameid", datatype=DataType.INT64)
        # schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)
        # schema.add_field(field_name="direction", datatype=DataType.VARCHAR, max_length=10)
        # schema.add_field(field_name="image_cropped_obj_path_saved", datatype=DataType.VARCHAR, max_length=255)
        # schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=FEAT_DIM)
        
        # index_params = client.prepare_index_params()
        # index_params.add_index(field_name="trackingId", index_name="trackletid_index")
        # index_params.add_index(field_name="frameid", index_name="frameid_index")
        # index_params.add_index(field_name="direction", index_name="direction_index")
        # index_params.add_index(field_name="embedding", index_name="embedding_index", metric_type="COSINE")
        
        # client.create_collection(
        #     collection_name=collection_name,
        #     schema=schema,
        #     index_params=index_params
        # )
        print(f"Not found collection: {collection_name}")
        exit(1)
    else:
        print(f"Collection {collection_name} already exists")

def get_exit_vectors(client: MilvusClient, collection_name: str) -> list[dict]:
    """Get all exit vectors from the collection"""
    filter_expr = "isRepresentative == True and direction == 'Exit'"
    
    results = client.query(
        collection_name=collection_name,
        filter=filter_expr,
        output_fields=["id", "timestamp", "trackingId", "embedding", "imgPath"],
        limit=10000  # Adjust based on your data size
    )
    
    # Sort by trackingId for consistent processing
    results.sort(key=lambda x: x['timestamp'])
    return results

def find_matching_enter(client: MilvusClient, collection_name: str, exit_timestamp: str, exit_embedding: list[float], used_enter_trackletids: set):
    """Find the best matching enter vector for an exit vector"""
    # Build filter expression
    filter_parts = ["isRepresentative == True","direction == 'Enter'", f"timestamp < '{exit_timestamp}'"]
    
    if used_enter_trackletids:
        excluded_ids = ', '.join(map(str, used_enter_trackletids))
        filter_parts.append(f"trackingId not in [{excluded_ids}]")
    
    filter_expr = " and ".join(filter_parts)
    
    # Perform vector search
    search_params = {
        "metric_type": "COSINE",
        "params": {}
    }
    
    try:
        results = client.search(
            collection_name=collection_name,
            data=[exit_embedding],
            anns_field="embedding",
            search_params=search_params,
            limit=1,
            filter=filter_expr,
            output_fields=["timestamp", "trackingId", "direction", "imgPath"]
        )
        
        if results and len(results[0]) > 0:
            match = results[0][0]
            return {
                'timestamp': match['entity']['timestamp'],
                'trackingId': match['entity']['trackingId'],
                'direction': match['entity']['direction'],
                'distance': match['distance'],
                'imgPath': match['entity'].get('imgPath', '')
            }
    except Exception as e:
        print(f"Error in vector search: {e}")
    
    return None

def process_exit_vectors(client: MilvusClient, collection_name : str, exit_vectors: list[dict], similarity_threshold: float):
    """Process all exit vectors and find matches"""
    results = []
    used_enter_trackletids = set()
    matched_enter_trackletids = set()
    
    for exit_vector in exit_vectors:
        exit_trackletid = exit_vector['trackingId']
        exit_timestamp = exit_vector['timestamp']
        exit_embedding = exit_vector['embedding']
        exit_image_path = exit_vector.get('imgPath', '')
        
        match = find_matching_enter(client, collection_name, exit_timestamp, exit_embedding, used_enter_trackletids)
        
        if match:
            similarity = abs(match['distance'])
            matched_trackletid = match['trackingId']
            matched_image_path = match.get('imgPath', '')
            
            if similarity >= similarity_threshold:
                category = "Inside"
                used_enter_trackletids.add(matched_trackletid)
                matched_enter_trackletids.add(matched_trackletid)
            else:
                category = "Ghost"
                matched_trackletid = None
                matched_image_path = None
            
            results.append({
                "exit_trackletid": exit_trackletid,
                "exit_image_path": exit_image_path,
                "matched_enter_trackletid": matched_trackletid,
                "matched_enter_image_path": matched_image_path,
                "similarity": similarity,
                "category": category
            })
        else:
            results.append({
                "exit_trackletid": exit_trackletid,
                "exit_image_path": exit_image_path,
                "matched_enter_trackletid": None,
                "matched_enter_image_path": None,
                "similarity": -1.0,
                "category": "Ghost"
            })
    
    return results, matched_enter_trackletids

def get_missing_enter_trackletids(client, collection_name, matched_enter_trackletids):
    """Get enter trackletids that were not matched"""
    filter_expr = "isRepresentative == True and direction == 'Enter'"
    
    results = client.query(
        collection_name=collection_name,
        filter=filter_expr,
        output_fields=["trackingId", "imgPath"],
        limit=10000
    )
    
    # Needs to return detailed missing objects
    all_enters = {result['trackingId']: result for result in results}
    missing_ids = set(all_enters.keys()) - matched_enter_trackletids
    
    missing_details = []
    for m_id in missing_ids:
        missing_details.append({
            "enter_trackletid": m_id,
            "enter_image_path": all_enters[m_id].get('imgPath', '')
        })
        
    return missing_ids, missing_details

def save_results_to_csv(results, missing_details, filename="matching_results.csv"):
    """Save results to CSV file"""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ExitTrackletID", "ExitImagePath", "EnterTrackletID", "EnterImagePath", "Similarity", "Category"])
        
        for res in results:
            writer.writerow([
                res['exit_trackletid'], 
                res['exit_image_path'],
                res['matched_enter_trackletid'], 
                res['matched_enter_image_path'],
                f"{res['similarity']:.4f}", 
                res['category']
            ])
        
        for missing in missing_details:
            writer.writerow([None, None, missing['enter_trackletid'], missing['enter_image_path'], None, "Missing"])

def run_exit_ranking(threshold=0.60, app_config=None):
    """Callable endpoint for the web UI that returns JSON results"""
    if app_config is None:
        app_config = load_config()
    client = setup_milvus_client(app_config.DEFAULT_SQL_DB)
    
    try:
        check_collection_if_not_exists(client, app_config.COLLECTION_NAME)
        exit_vectors = get_exit_vectors(client, app_config.COLLECTION_NAME)
        results, matched_enter_trackletids = process_exit_vectors(
            client, app_config.COLLECTION_NAME, exit_vectors, threshold
        )
        missing_enter_trackletids, missing_details = get_missing_enter_trackletids(
            client, app_config.COLLECTION_NAME, matched_enter_trackletids
        )
        
        save_results_to_csv(results, missing_details)
        
        inside_count = sum(1 for r in results if r['category'] == 'Inside')
        ghost_count = sum(1 for r in results if r['category'] == 'Ghost')
        missing_count = len(missing_enter_trackletids)
        
        return {
            "status": "success",
            "summary": {
                "inside": inside_count,
                "ghost": ghost_count,
                "missing": missing_count
            },
            "matches": results,
            "missing": missing_details
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        client.close()

def main():
    args = parse_arguments()
    res = run_exit_ranking(args.threshold)
    if res["status"] == "success":
        print(f"Results saved. Summary: Inside={res['summary']['inside']}, Ghost={res['summary']['ghost']}, Missing={res['summary']['missing']}")
    else:
        print(f"Failed: {res['message']}")

if __name__ == "__main__":
    main()