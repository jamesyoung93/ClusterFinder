
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from matplotlib.colors import LinearSegmentedColormap

# Load the combined DataFrame
@st.cache_data
def load_data():
    df = pd.read_csv("combined_prepared.csv")
    return df

df = load_data()

# Remove commas from 'ESTIMATED_VALUE' and ensure it is numeric
df['ESTIMATED_VALUE'] = df['ESTIMATED_VALUE'].astype(str).str.replace(',', '').astype(float).astype(int)

# Ensure 'YEAR' is numeric
df['YEAR'] = df['YEAR'].astype(str).str.replace(',', '').astype(float).astype(int)

# Filter permit types to only include 0 and 1
permit_columns = [
    'BUILDING_COMMERCIAL', 'BUILDING_RESIDENTIAL', 'ELECTRICAL_COMMERCIAL',
    'ELECTRICAL_RESIDENTIAL', 'MECHANICAL_COMMERCIAL', 'MECHANICAL_RESIDENTIAL',
    'PLUMBING_COMMERCIAL', 'PLUMBING_RESIDENTIAL'
]

for col in permit_columns:
    df[col] = df[col].apply(lambda x: x if x in [0, 1] else 0)

# Get the list of cities
cities = df['City'].unique()

# Sidebar for user inputs
st.sidebar.header('Filter Options')

selected_city = st.sidebar.selectbox('Select City', cities)

# Filter the DataFrame based on the selected city
city_df = df[df['City'] == selected_city]

# Identify permit types
permit_types = [col for col in permit_columns if col in city_df.columns and city_df[col].isin([0, 1]).all()]

if len(permit_types) == 0:
    st.error("No permit types found in the selected city's dataset.")
    st.stop()

selected_permits = st.sidebar.multiselect('Select Permit Type(s)', permit_types, default=permit_types[0])

# Ensure selected permits exist in DataFrame
for permit in selected_permits:
    if permit not in city_df.columns:
        st.error(f"The permit type '{permit}' does not exist in the dataset.")
        st.stop()

# Check if 'YEAR' column exists
if 'YEAR' not in city_df.columns:
    st.error("The column 'YEAR' does not exist in the dataset.")
    st.stop()

# Filter by year range
years = city_df['YEAR'].unique()
if len(years) == 0:
    st.error("No year data found in the selected city's dataset.")
    st.stop()

year_min = int(years.min())
year_max = int(years.max())

year_range = st.sidebar.slider(
    'Select Year Range',
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

# Filter the DataFrame based on the selected year range
city_df = city_df[(city_df['YEAR'] >= year_range[0]) & (city_df['YEAR'] <= year_range[1])]

# Check if 'Longitude' and 'Latitude' columns exist
if 'Longitude' not in city_df.columns or 'Latitude' not in city_df.columns:
    st.error("The columns 'Longitude' and 'Latitude' do not exist in the dataset.")
    st.stop()

# Assuming 'latitude' and 'longitude' are not in the DataFrame, we'll reproject them
geometry = [Point(xy) for xy in zip(city_df['Longitude'], city_df['Latitude'])]
gdf = gpd.GeoDataFrame(city_df, geometry=geometry, crs="EPSG:4326")  # Now using WGS84 directly

# Add new columns for latitude and longitude
city_df['longitude'] = gdf.geometry.x
city_df['latitude'] = gdf.geometry.y

# Filter by longitude and latitude range
lon_min = float(city_df['longitude'].min())
lon_max = float(city_df['longitude'].max())
lat_min = float(city_df['latitude'].min())
lat_max = float(city_df['latitude'].max())

lon_range = st.sidebar.slider(
    'Select Longitude Range',
    min_value=lon_min,
    max_value=lon_max,
    value=(lon_min, lon_max)
)

lat_range = st.sidebar.slider(
    'Select Latitude Range',
    min_value=lat_min,
    max_value=lat_max,
    value=(lat_min, lat_max)
)

# Filter the DataFrame based on the selected longitude and latitude range
city_df = city_df[(city_df['longitude'] >= lon_range[0]) & (city_df['longitude'] <= lon_range[1]) &
                  (city_df['latitude'] >= lat_range[0]) & (city_df['latitude'] <= lat_range[1])]

# Aggregate selected permit types
city_df['Permit_Sum'] = city_df[selected_permits].sum(axis=1)

# Select only the necessary columns
city_df = city_df[['YEAR', 'latitude', 'longitude', 'Permit_Sum', #'CONTRACTOR_NAME', 'ESTIMATED_VALUE'] + selected_permits].copy()

# Create bins for latitude and longitude
cell_size = st.sidebar.number_input('Cell Size', value=0.01, step=0.01, min_value = 0.01)

x_bins = np.arange(city_df['longitude'].min(), city_df['longitude'].max() + cell_size, cell_size)
y_bins = np.arange(city_df['latitude'].min(), city_df['latitude'].max() + cell_size, cell_size)

city_df['X Bin'] = pd.cut(city_df['longitude'], bins=x_bins, labels=x_bins[:-1], include_lowest=True)
city_df['Y Bin'] = pd.cut(city_df['latitude'], bins=y_bins, labels=y_bins[:-1], include_lowest=True)

bin_stats = city_df.groupby(['X Bin', 'Y Bin'], observed=True).agg(
    Permit_Sum=('Permit_Sum', 'sum'),
    Count=('Permit_Sum', 'size')
).reset_index()

# Drop NaN values
bin_stats = bin_stats.dropna(subset=['X Bin', 'Y Bin'])

# Convert bins to float
bin_stats['X Bin'] = bin_stats['X Bin'].astype(float)
bin_stats['Y Bin'] = bin_stats['Y Bin'].astype(float)

# Create GeoDataFrame
geometry = [Point(xy) for xy in zip(bin_stats['X Bin'], bin_stats['Y Bin'])]
gdf = gpd.GeoDataFrame(bin_stats, geometry=geometry, crs="EPSG:4326")

# Custom color map similar to the provided image
colors = ["#440154", "#30678D", "#35B779", "#FDE724"]
cmap_name = 'custom_cmap'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

# Create a map plot with contextily basemap
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Convert the heatmap to the same CRS as the basemap
xmin, ymin, xmax, ymax = gdf.total_bounds
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Add basemap
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Add the scatter plot with custom color map
scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y, c=gdf['Permit_Sum'], cmap=cm, alpha=0.6, s=50, edgecolor='k', zorder=2)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Sum of Selected Permits')

# Plot adjustments
plt.title(f'Selected Permits in {selected_city} ({year_range[0]} - {year_range[1]}) with Basemap', fontsize=18)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)

# Display the plot in Streamlit
st.pyplot(fig)

# Aggregate permit columns to count jobs
city_df['Job_Count'] = city_df[selected_permits].sum(axis=1)

# Filter the city_df based on selected permits only
filtered_city_df = city_df[city_df['Job_Count'] > 0]

# Plot total number of jobs or total value of jobs over the years
st.subheader("Total Number of Jobs or Total Value of Jobs Over the Years")

fig, ax = plt.subplots()

# Plot for each selected permit type
for permit in selected_permits:
    yearly_data = filtered_city_df.groupby('YEAR')[permit].sum()
    ax.plot(yearly_data.index, yearly_data.values, label=permit)

ax.set_xlabel('Year')
ax.set_ylabel('Total Number of Jobs')
ax.set_title('Total Number of Jobs Over the Years')
ax.legend()
plt.figtext(0.5, 0.01, "Note: The data for the current year, 2024, is not complete.", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

st.pyplot(fig)




