import earthaccess
import boto3
import sys
import warnings
from datetime import datetime

# Suppress the earthaccess FutureWarnings to keep your console clean
warnings.filterwarnings("ignore", category=FutureWarning)
class ProgressPercentage(object):
    """A simple callback class to show upload progress."""
    def __init__(self, filename):
        self._filename = filename
        self._seen_so_far = 0

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        # Print progress in MB on the same line
        sys.stdout.write(
            f"\r    -> Uploading {self._filename}: {self._seen_so_far / (1024 * 1024):.2f} MB transferred"
        )
        sys.stdout.flush()
        
# 1. Authenticate with NASA Earthdata Login
# This will look for credentials in environment variables or your ~/.netrc file.
# If neither is found, it will prompt you for your Earthdata username and password.
earthaccess.login()

# 2. Define search parameters
short_name = "VJ110A1F"
# Bounding box format: (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)
bounding_box = (-116.1, 35.32, -89.46, 50.59)
#date_range = ("2019-01-01", "2019-01-07")
date_range = ("2019-01-08", "2019-07-01")

target_bucket = "airborne-smce-prod-user-bucket"
s3_client = boto3.client('s3')
base_prefix = f"JOIN/VIIRS/{short_name}"

print(f"Searching for {short_name} granules...")

# 3. Search for the granules
results = earthaccess.search_data(
    short_name=short_name,
    bounding_box=bounding_box,
    temporal=date_range   
)

print(f"Found {len(results)} granules.\n")

# # 4. Extract S3 URIs (direct access links)
# s3_uris = []
# for granule in granules:
#     # "direct" access gives you the s3:// links instead of HTTPS
#     links = granule.data_links(access="direct")
#     s3_uris.extend(links)

# print("Available S3 URIs:")
# for uri in s3_uris:
#     print(uri)

# # 5. Accessing the data (Must be in AWS us-west-2)
# # If you are running this code on an EC2 instance or SageMaker in AWS us-west-2,
# # you can use earthaccess.open() to read the S3 files directly into memory (e.g., with xarray).
# try:
#     # earthaccess.open handles the temporary S3 credentials automatically
#     file_objects = earthaccess.open(granules)
#     print(f"\nSuccessfully opened {len(file_objects)} files via S3!")
    
#     # Example: you could now pass these file_objects to xarray
#     # import xarray as xr
#     # ds = xr.open_dataset(file_objects[0])
    
# except Exception as e:
#     print("\nCould not open files directly via S3. Are you running this in AWS us-west-2?")
#     print(f"Error: {e}")

# 4. Stream directly to destination S3 (No local files created)
for granule in results:
    # Extract the start date from the metadata to format the subdirectory
    start_time_str = granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
    dt_obj = datetime.strptime(start_time_str[:10], "%Y-%m-%d")
    date_path = dt_obj.strftime("%Y/%m/%d")

    # Get the filename from the first data link
    data_links = granule.data_links()
    if not data_links:
        continue
    filename = data_links[0].split('/')[-1]

    # Construct the S3 key
    s3_key = f"{base_prefix}/{date_path}/{filename}"
    print(f"\nStreaming: {filename} \nTo: s3://{target_bucket}/{s3_key}")

    # Open the file as a streaming fsspec object
    file_objs = earthaccess.open([granule])
    
    if not file_objs:
        print(f"Failed to open stream for {filename}. Skipping.")
        continue
        
    file_obj = file_objs[0]

    try:
        # upload_fileobj reads the stream in chunks and uploads it directly to S3
        # The Callback shows live progress
        s3_client.upload_fileobj(
            file_obj, 
            target_bucket, 
            s3_key,
            Callback=ProgressPercentage(filename)
        )
        print("\n    -> Upload complete.")
    except Exception as e:
        print(f"\n    -> Error uploading {filename} to S3: {e}")
    finally:
        # Ensure the stream is closed
        file_obj.close()

print("\nTransfer process complete.")