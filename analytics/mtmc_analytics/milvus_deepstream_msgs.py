import argparse
import os
import threading
import time
from kafka import KafkaConsumer
from json import loads, dumps
import uuid
from pymilvus import MilvusClient
from mtmc_reid.database.tracklet_schema import TrackletRecord
from mtmc_reid.database.milvus_schema_init import init_db
from pydantic import ValidationError
from mtmc_reid.configs.app_config import load_config
from typing import List,Dict,Any
config = load_config()
milvus_db_path = config.DEFAULT_SQL_DB
collection_name = config.COLLECTION_NAME
kakfa_topic_name = config.KAFKA_TOPIC
kafka_msg_log = 'messages-all-renew.json'
# stop_signal = threading.Event()

def process_event(event_data, json_file, client):
    """Process a single Kafka event and write data to files."""
    json_file.write(dumps(event_data) + '\n')
    version = event_data.get("version", "")
    frameid = int(event_data.get("id", ""))
    timestamp = event_data.get("@timestamp", "")
    sensorId = event_data.get("sensorId", "")
    if sensorId in config.ENTER_SENSORIDS:
        direction = "Enter"
    elif sensorId in config.EXIT_SENSORIDS:
        direction = "Exit"
    else:
        print(f"No valid sensorIds list in event with id={frameid}")
        return
    objects = event_data.get("objects")
    if not isinstance(objects, list):
        print(f"No valid 'objects' list in event with id={frameid}")
        return

    for obj in objects:
        # check generate_event_message_minimal from deepstream-7.1/sources/libs/nvmsgconv/deepstream_schema/eventmsg_payload.cpp
        if isinstance(obj, str):
            try:
                parts = obj.split('|') 
                trackingId = int(parts[0])
                bbox_left = float(parts[1])
                bbox_top = float(parts[2])
                bbox_right = float(parts[3])
                bbox_bottom = float(parts[4])
                objClassName = parts[5]
                imgPath = parts[9]
                confidence = float(parts[12])
                embedding = parts[15]
                bbox_width = bbox_right - bbox_left
                bbox_height = bbox_bottom - bbox_top
                embedding = [float(num) for num in embedding.split(',')]
                payload = TrackletRecord.model_validate({"version": version,
                        "frameid": frameid,
                        "timestamp": timestamp,
                        "sensorId":sensorId,
                        "trackingId": trackingId,
                        "direction" : direction,
                        "confidence": confidence,
                        "bbox_top":bbox_top,
                        "bbox_left":bbox_left,
                        "bbox_width":bbox_width,
                        "bbox_height":bbox_height,
                        "imgPath":imgPath,
                        "objClassName":objClassName,
                        "embedding": embedding},context={"config":config})
                payload_data = payload.model_dump()
                res = client.insert(
                    collection_name=collection_name,
                    data=payload_data
                )
                print(f"{res}\n")
            except ValidationError as e:
                print(f"{e}\n")
                
            except Exception as e:
                print(f"An error occurred: {e}\n")
                print(f"{payload.model_dump_json()}\n")
                
        else:
            print(f"No valid 'obj' string in event with id={frameid}")
            return
def extract_payloads(event_data) -> List[Dict[str, Any]]:
    """Extract valid TrackletRecord payloads from a single event."""
    payloads = []
    version = event_data.get("version", "")
    frameid = int(event_data.get("id", ""))
    timestamp = event_data.get("@timestamp", "")
    sensorId = event_data.get("sensorId", "")

    if sensorId in config.ENTER_SENSORIDS:
        direction = "Enter"
    elif sensorId in config.EXIT_SENSORIDS:
        direction = "Exit"
    else:
        return payloads  # Invalid sensor ID

    objects = event_data.get("objects", [])
    for obj in objects:
        try:
            if not isinstance(obj, str):
                continue
            parts = obj.split('|')
            trackingId = int(parts[0])
            bbox_left = float(parts[1])
            bbox_top = float(parts[2])
            bbox_right = float(parts[3])
            bbox_bottom = float(parts[4])
            objClassName = parts[5]
            imgPath = parts[9]
            confidence = float(parts[12])
            embedding = [float(num) for num in parts[15].split(',')]
            bbox_width = bbox_right - bbox_left
            bbox_height = bbox_bottom - bbox_top

            payload = TrackletRecord.model_validate({
                "version": version,
                "frameid": frameid,
                "timestamp": timestamp,
                "sensorId": sensorId,
                "trackingId": trackingId,
                "direction": direction,
                "confidence": confidence,
                "bbox_top": bbox_top,
                "bbox_left": bbox_left,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "imgPath": imgPath,
                "objClassName": objClassName,
                "embedding": embedding
            }, context={"config": config})
            payloads.append(payload.model_dump())

        except ValidationError as e:
            print(f"Validation error: {e}")
        except Exception as e:
            print(f"Parsing error: {e}")
    return payloads
# def wait_for_shutdown_signal_from_pipe(pipe_path="/tmp/stop_pipe"):
#     try:
#         with open(pipe_path, "r") as pipe:
#             for line in pipe:
#                 if line.strip().lower() in ("exit", "quit"):
#                     stop_signal.set()
#                     break
#     except Exception as e:
#         print(f"Pipe read error: {e}")

# Argument parser to receive output directory
parser = argparse.ArgumentParser(description="Consume Kafka messages and save to files in a directory.")
parser.add_argument('--output_dir', help='Directory to save output files', default="./")
args = parser.parse_args()

# Ensure the output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Define output file paths
all_file_path = os.path.join(args.output_dir, kafka_msg_log)

init_db(config)
client = MilvusClient(milvus_db_path)
res = client.list_collections()

if collection_name not in res:
    print(f'{collection_name} does not exist in {milvus_db_path}')
    client.close()
    exit(1)

consumer = KafkaConsumer(
    kakfa_topic_name,
    bootstrap_servers='127.0.0.1:9092',
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id="milvus-deepstream-consumer",
    value_deserializer=lambda x: loads(x.decode('utf-8'))
)
# input_thread = threading.Thread(target=wait_for_shutdown_signal_from_pipe, daemon=True)
# input_thread.start()
try:

    print("Waiting for Kafka partition assignment...")
    consumer.poll(timeout_ms=1000)
    while not consumer.assignment():
        time.sleep(0.5)
        consumer.poll(timeout_ms=1000)

    print(f"Partitions assigned: {consumer.assignment()}")
    consumer.seek_to_end()

    BATCH_SIZE = 1
    buffer = []

    # Open output file and consume messages
    with open(all_file_path, 'w') as json_file:
        for event in consumer:
            event_data = event.value
            json_file.write(dumps(event_data) + '\n')
            
            try:
                new_payloads = extract_payloads(event_data)
                buffer.extend(new_payloads)
            except Exception as e:
                print(f"Error extracting payloads: {e}")
            
            # Batch insert condition
            if len(buffer) >= BATCH_SIZE:
                try:
                    client.insert(collection_name=collection_name, data=buffer)
                    print(f"Inserted batch of {len(buffer)}")
                    buffer.clear()
                except Exception as e:
                    print(f"Batch insert error: {e}")
                    buffer.clear()  # Optionally keep for retry logic
            # if stop_signal.is_set():
            #     print("Shutdown signal received from stdin.")
            #     if buffer:
            #         try:
            #             client.insert(collection_name=collection_name, data=buffer)
            #         except Exception as e:
            #             print(f"Final batch insert failed: {e}")
            #             # Fallback: Dump to JSON file for retry later
            #             fallback_path = os.path.join(args.output_dir, "failed_insert_backup.json")
            #             with open(fallback_path, "w") as f:
            #                 for item in buffer:
            #                     f.write(dumps(item) + "\n")
            #             print(f"Saved failed insert batch to: {fallback_path}")
            #     break
except KeyboardInterrupt:
    print("Interrupted by user. Shutting down...")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Closing consumer and client.")
    consumer.close()
    client.close()

