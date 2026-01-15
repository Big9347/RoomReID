from pymilvus import MilvusClient, DataType
from mtmc_reid.configs.app_config import load_config,AppConfig

def init_db(config: AppConfig):
    if not config:
        print("No appconfig to init db")
        exit(1)
    TABLE_NAME = config.COLLECTION_NAME
    DIM = config.REID_FEATURE_DIM
    client = MilvusClient(config.DEFAULT_SQL_DB)
    if TABLE_NAME  in client.list_collections():
        print(f"{TABLE_NAME} already exist in {config.DEFAULT_SQL_DB}")
        return
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=False
    )


    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary =True)
    schema.add_field(field_name="version", datatype=DataType.VARCHAR, max_length=5)
    schema.add_field(field_name="frameid", datatype=DataType.INT64)
    schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=25)
    schema.add_field(field_name="sensorId", datatype=DataType.VARCHAR, max_length=25)
    schema.add_field(field_name="trackingId", datatype=DataType.INT16)
    schema.add_field(field_name="direction", datatype=DataType.VARCHAR, max_length=5)
    schema.add_field(field_name="confidence", datatype=DataType.FLOAT)
    schema.add_field(field_name="bbox_top", datatype=DataType.FLOAT)
    schema.add_field(field_name="bbox_left", datatype=DataType.FLOAT)
    schema.add_field(field_name="bbox_width", datatype=DataType.FLOAT)
    schema.add_field(field_name="bbox_height", datatype=DataType.FLOAT)
    schema.add_field(field_name="imgPath", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="objClassName", datatype=DataType.VARCHAR, max_length=25)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field(field_name="isTransit", datatype=DataType.BOOL)
    schema.add_field(field_name="isRepresentative", datatype=DataType.BOOL)


    schema.verify()

    print(schema)
    

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="trackingId",index_name="trackingId_index")
    index_params.add_index(field_name="frameid",index_name="frameid_index")
    index_params.add_index(field_name="timestamp",index_name="timestamp_index")
    index_params.add_index(field_name="embedding",index_name="embedding_index",metric_type="COSINE")
    try:
        client.create_collection(
        collection_name=TABLE_NAME,
        schema=schema,
        index_params=index_params
        )
        res = client.describe_collection(
        collection_name=TABLE_NAME
        )
        print(res)
    except Exception as e:
        print(e)
        exit(1)
    finally:
        client.close()
        
    

if __name__ == "__main__":
    init_db(load_config())