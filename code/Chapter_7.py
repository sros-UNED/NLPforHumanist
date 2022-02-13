"""
Humanities Data Analysis: Case studies with Python
--------------------------------------------------
Folgert Karsdorp, Mike Kestemont & Allen Riddell
Chapter 7: Narrating with Maps
"""


# %%
import pandas as pd

df = pd.read_csv('data/cwsac_battles.csv', parse_dates=['start_date'], index_col=0)
df.loc[df['battle_name'].str.contains('Cold Harbor')].T


# %%
import pickle
import operator


with open('data/cwsac_battles_locations.pkl', 'rb') as f:
    locations = pickle.load(f)

# first, exclude 2 battles (of 384) not associated with named locations
df = df.loc[df['locations'].notnull()]

# second, extract the first place name associated with the battle.
# (Battles which took place over several days were often associated
# with multiple (nearby) locations.)
df['location_name'] = df['locations'].str.split(';').apply(operator.itemgetter(0))

# finally, add latitude (`lat`) and longitude ('lon') to each row
df['lat'] = df['location_name'].apply(lambda name: locations[name]['lat'])
df['lon'] = df['location_name'].apply(lambda name: locations[name]['lon'])


# %%
columns_of_interest = [
    'battle_name', 'locations', 'start_date', 'casualties', 'lat', 'lon', 'result',
    'campaign'
]
df[columns_of_interest].sort_values('casualties', ascending=False).head(3)


# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
from cartopy.feature import ShapelyFeature

# Step 1: Define the desired projection with appropriate parameters
# Lambert Conformal Conic (LCC) is a recommended default projection.
# We use the parameters recommended for the United States described
# at http://www.georeference.org/doc/lambert_conformal_conic.htm :
# 1. Center of lower 48 states is roughly 38°, -100°
# 2. 32° for first standard latitude and 44° for the second latitude

projection = ccrs.LambertConformal(
    central_latitude=38, central_longitude=-100,
    standard_parallels=(32, 44))

# Step 2: Set up a base figure and attach a subfigure with the
# defined projection
fig = plt.figure(figsize=(8, 8))
m = fig.add_subplot(1, 1, 1, projection=projection)
# Limit the displayed to a bounding rectangle
m.set_extent([-70, -100, 40, 25], crs=ccrs.PlateCarree())

# Step 3: Read the shapefile and transform it into a ShapelyFeature
shape_feature = ShapelyFeature(
    shapereader.Reader('data/st99_d00.shp').geometries(),
    ccrs.PlateCarree(), facecolor='lightgray', edgecolor='white')
m.add_feature(shape_feature, linewidth=0.3)

# Step 4: Add some aesthetics (i.e. no outline box)
m.outline_patch.set_visible(False)


# %%
def basemap(shapefile, projection, extent=None, nrows=1, ncols=1, figsize=(8, 8)):
    f, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=100,
        subplot_kw=dict(projection=projection, frameon=False))
    axes = [axes] if (nrows + ncols) == 2 else axes.flatten()
    for ax in axes:
        ax.set_extent(extent, ccrs.PlateCarree())
        shape_feature = ShapelyFeature(
            shapereader.Reader(shapefile).geometries(),
            ccrs.PlateCarree(), facecolor='lightgray', edgecolor='white')
        ax.add_feature(shape_feature, linewidth=0.3)
        ax.outline_patch.set_visible(False)
    return f, (axes[0] if (nrows + ncols) == 2 else axes)

def civil_war_basemap(nrows=1, ncols=1, figsize=(8, 8)):
    projection = ccrs.LambertConformal(
        central_latitude=38, central_longitude=-100,
        standard_parallels=(32, 44))
    extent = -70, -100, 40, 25
    return basemap('data/st99_d00.shp', projection, extent=extent,
                   nrows=nrows, ncols=ncols, figsize=figsize)


# %%
# Richmond, Virginia has decimal latitude and longitude:
#    37.533333, -77.466667
x, y = m.transData.transform((37.533333, -77.466667))
print(x, y)


# %%
# Recover the latitude and longitude for Richmond, Virginia
print(m.transData.inverted().transform((x, y)))


# %%
battles_of_interest = ['LA003', 'KY008', 'VA026']
three_battles = df.loc[battles_of_interest]


# %%
battle_names = three_battles['battle_name']
battle_years = three_battles['start_date'].dt.year
labels = [f'{name} ({year})' for name, year in zip(battle_names, battle_years)]
print(labels)


# %%
# draw the map
f, m = civil_war_basemap(figsize=(8, 8))
# add points
m.scatter(three_battles['lon'], three_battles['lat'],
          zorder=2, marker='o', alpha=0.7, transform=ccrs.PlateCarree())
# add labels
for x, y, label in zip(three_battles['lon'], three_battles['lat'], labels):
    # NOTE: the "plt.annotate call" does not have a "transform=" keyword,
    # so for this one we transform the coordinates with a Cartopy call.
    x, y = m.projection.transform_point(x, y, src_crs=ccrs.PlateCarree())
    # position each label to the right of the point
    # give the label a semi-transparent background so it is easier to see
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(10, 0),
                 # xytext is measured in figure points,
                 # 0.353 mm or 1/72 of an inch
                 textcoords='offset points',
                 bbox=dict(fc='#f2f2f2', alpha=0.7))


# %%
three_battles[columns_of_interest]


# %%
import itertools
import matplotlib.cm


def result_color(result_type):
    """Helper function: return a qualitative color for each
       party in the war.
    """
    result_types = 'Confederate', 'Inconclusive', 'Union'
    # qualitative color map, suited for categorical data
    color_map = matplotlib.cm.tab10
    return color_map(result_types.index(result_type))


def plot_battles(lat, lon, casualties, results, m=None, figsize=(8, 8)):
    """Draw circles with area proportional to `casualties`
       at `lat`, `lon`.
    """
    if m is None:
        f, m = civil_war_basemap(figsize=figsize)
    else:
        f, m = m
    # make a circle proportional to the casualties
    # divide by a constant, otherwise circles will cover much of the map
    size = casualties / 50
    for result_type, result_group in itertools.groupby(
        zip(lat, lon, size, results), key=operator.itemgetter(3)):
        lat, lon, size, results = zip(*list(result_group))
        color = result_color(result_type)
        m.scatter(lon, lat, s=size, color=color, alpha=0.8,
                  label=result_type, transform=ccrs.PlateCarree(), zorder=2)
    return f, m


# %%
lat, lon, casualties, results = (three_battles['lat'], three_battles['lon'],
                                 three_battles['casualties'], three_battles['result'])
plot_battles(lat, lon, casualties, results)
plt.legend(loc='upper left');


# %%
df.loc[df['casualties'].isnull(), columns_of_interest].head(3)


# %%
print(df['casualties'].quantile(0.20))


# %%
df.loc[df['casualties'].isnull(), 'casualties'] = df['casualties'].quantile(0.20)


# %%
df.groupby(df['start_date'].dt.year).size()
# alternatively, df['start_date'].dt.year.value_counts()


# %%
df.groupby(df['start_date'].dt.year)['casualties'].sum()


# %%
df.groupby(df.start_date.dt.strftime('%Y-%m'))['casualties'].sum().head()


# %%
import calendar
import itertools

f, maps = civil_war_basemap(nrows=5, ncols=12, figsize=(18, 8))

# Predefine an iterable of dates. The war begins in April 1861, and
# the Confederate government dissolves in spring of 1865.
dates = itertools.product(range(1861, 1865 + 1), range(1, 12 + 1))

for (year, month), m in zip(dates, maps):
    battles = df.loc[(df['start_date'].dt.year == year) &
                     (df['start_date'].dt.month == month)]
    lat, lon = battles['lat'].values, battles['lon'].values
    casualties, results = battles['casualties'], battles['result']
    plot_battles(lat, lon, casualties, results, m=(f, m))
    month_abbrev = calendar.month_abbr[month]
    m.set_title(f'{year}-{month_abbrev}')

plt.tight_layout();


# %%
def denmark_basemap(nrows=1, ncols=1, figsize=(8, 8)):
    projection = ccrs.LambertConformal(central_latitude=50, central_longitude=10)
    extent = 8.09, 14.15, 54.56, 57.75
    return basemap('data/denmark/denmark.shp', projection, extent=extent,
                   nrows=nrows, ncols=ncols, figsize=figsize)

fig, m = denmark_basemap()


