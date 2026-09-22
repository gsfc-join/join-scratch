import os
import boto3
import icechunk
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from urllib.parse import urlparse

def setup_aws_credentials():
    """Use boto3 to grab the working credentials and set them for obstore."""
    session = boto3.Session()
    creds = session.get_credentials()

    if creds:
        frozen_creds = creds.get_frozen_credentials()
        os.environ["AWS_ACCESS_KEY_ID"] = frozen_creds.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = frozen_creds.secret_key
        if frozen_creds.token:
            os.environ["AWS_SESSION_TOKEN"] = frozen_creds.token
            
        # Prevent obstore from trying to hit the EC2 metadata endpoint
        os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
        
        # SMCE environments are typically us-west-2, but we'll grab it from the session if available.
        if session.region_name:
            os.environ["AWS_REGION"] = session.region_name
            
        print("Successfully injected boto3 credentials into environment.")
    else:
        print("Warning: boto3 could not find AWS credentials.")

def plot_global_swe(s3_url, target_date="2019-01-01", branch_name="main"):
    print(f"Connecting to Icechunk repository at: {s3_url}")
    
    try:
        # Parse the S3 URL
        parsed = urlparse(s3_url)
        bucket_name = parsed.netloc
        prefix_path = parsed.path.lstrip("/")
        
        # 1. Create the Storage object
        storage = icechunk.s3_storage(
            bucket=bucket_name,
            prefix=prefix_path,
            region=os.environ.get("AWS_REGION", "us-west-2") 
        )
        
        # 2. Open the Icechunk repository and session
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session(branch=branch_name)
        
        # 3. Open the dataset
        ds = xr.open_zarr(session.store, consolidated=False)
        print("Dataset successfully loaded.")
        
        # 4. Identify the time coordinate dynamically
        time_coord = None
        for potential_name in ['time', 'date', 'datetime', 't']:
            if potential_name in ds.coords:
                time_coord = potential_name
                break
                
        if not time_coord:
            raise ValueError(f"Could not find a time coordinate. Available coords: {list(ds.coords.keys())}")
            
        # 5. Identify the SWE variable (Checking common names for SWE)
        swe_var = None
        for potential_var in ['swe', 'SWE', 'Snow_Water_Equivalent', 'snow_water_equivalent']:
            if potential_var in ds.data_vars:
                swe_var = potential_var
                break
                
        if not swe_var:
            print(f"Warning: Could not automatically detect SWE variable. Available variables: {list(ds.data_vars.keys())}")
            # Fallback: manually prompt or use the first data variable (uncomment next line if you want to force it)
            # swe_var = list(ds.data_vars.keys())[0]
            return
            
        # 6. Select the specific date using xarray's .sel()
        print(f"Selecting data for {target_date}...")
        try:
            # Using method='nearest' in case the exact time is off by a few hours (e.g., 2019-01-01T12:00)
            ds_subset = ds.sel({time_coord: target_date}, method="nearest") 
        except KeyError:
            print(f"Date {target_date} not found or selection failed.")
            return

        # Extract the 2D DataArray for plotting
        swe_data = ds_subset[swe_var]
        
        # 7. Generate the Global Plot using Cartopy
        print("Generating global map...")
        fig = plt.figure(figsize=(15, 8))
        
        # Create a Cartopy PlateCarree projection axis
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines(linewidth=1.0, color='black')
        ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        
        # Plot the data
        # Note: xarray's .plot() wrapper works well with Cartopy if you pass the transform
        swe_data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='Blues',            # Blues colormap is standard for snow/water
            cbar_kwargs={'label': 'Snow Water Equivalent', 'shrink': 0.7, 'pad': 0.05},
            robust=True              # Robust calculates vmin/vmax using 2nd and 98th percentiles to handle outliers
        )
        
        actual_time = pd.to_datetime(ds_subset[time_coord].values).strftime('%Y-%m-%d %H:%M')
        plt.title(f"Global Snow Water Equivalent (SWE) Data Distribution\nDate: {actual_time}")
        plt.tight_layout()
        
        # Save or display
        plt.savefig("SWE_Distribution_20190101.png", dpi=300, bbox_inches='tight')
        print("Plot saved as 'SWE_Distribution_20190101.png'")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Setup the environment credentials first
    setup_aws_credentials()
    
    # Define the S3 path
    store_url = "s3://airborne-smce-prod-user-bucket/JOIN/icechunk-stores/CEDA_Store_v2/"
    
    # Execute the plotting function
    plot_global_swe(store_url, target_date="2019-01-01")