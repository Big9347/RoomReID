from pymilvus import MilvusClient
import datetime
from mtmc_reid.configs.app_config import load_config
from mtmc_reid.database.tracklet_schema import TrackletRecord
def iso_to_unix(iso_str):
    dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.timestamp()

import os
config = load_config()
client = MilvusClient(config.DEFAULT_SQL_DB,token=os.getenv("MILVUS_TOKEN", "root:Milvus"))
# res = client.query(
#     collection_name="my_collection",
#     filter=f'timestamp > "2025-03-06T05:41:40.720Z" and frameid < 1000000',
#     output_fields=["timestamp", "frameid"],
# )
# print(iso_to_unix("2025-03-06T05:30:15.720Z"), iso_to_unix("2025-03-06T09:32:15.720Z"))
# print(res)
#print(list(TrackletRecord.model_fields.keys()))
# res = client.query(
#                     collection_name=config.COLLECTION_NAME,
#                     filter='direction == "Enter"',
#                     output_fields=['trackingId', 'frameid', 'direction',
#                                     'bbox_top', 'bbox_left', 'bbox_width', 'bbox_height']
#                     )
#tr = [TrackletRecord.model_validate(record,context={"config":config}) for record in res]
# res = client.query(
#                     collection_name=config.COLLECTION_NAME,
#                     filter='',
#                     output_fields=['trackingId', 'frameid', 'direction',
#                                     'isTransit', 'isRepresentative'],
#                     limit =1000
#                     )

print(client.get_collection_stats(collection_name=config.COLLECTION_NAME))
client.close()
