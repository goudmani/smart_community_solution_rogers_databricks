# Databricks notebook source
# MAGIC %md
# MAGIC # Rogers × Databricks — Cell Tower Traffic EDA
# MAGIC **Goal:** understand transit flows and security-relevant crowd patterns before designing the smart-community solution.
# MAGIC
# MAGIC Columns: `location_name, longitude, latitude, timestamp, origin, dwell_time`
# MAGIC
# MAGIC All heavy aggregation stays in Spark; only small aggregated results go to pandas for plotting.
# MAGIC
# MAGIC | Section | What it answers |
# MAGIC |---|---|
# MAGIC | 0 | Setup + feature derivation |
# MAGIC | 1 | Data overview & quality |
# MAGIC | 2 | Synthetic-data artefact checks |
# MAGIC | 3 | Volume by location & origin |
# MAGIC | 4 | Dwell-time distribution |
# MAGIC | 5 | Temporal patterns |
# MAGIC | 6 | Spatial patterns (maps) |
# MAGIC | 7 | Origin → destination flows |
# MAGIC | 8 | Dwell behaviour by place & time |
# MAGIC | 9 | Crowd occupancy (people present) |
# MAGIC | 10 | Security / anomaly signals |
# MAGIC | 11 | Correlation & similarity |
# MAGIC | 12 | Location KPI table (saved to Delta) |

# COMMAND ----------

# MAGIC %md ## 0. Setup

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import Window
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
TABLE = "main.default.cell_tower_traffic"   # <-- change to your table
OUT_SCHEMA = "main.default"                 # <-- where the KPI table gets written
LOCAL_TZ = "America/Vancouver"              # timestamps end in Z (UTC); analyse in local time
TOP_N = 15                                  # locations shown in busy charts
DWELL_UNIT_SECONDS = 60                     # 60 if dwell_time is minutes, 1 if seconds — confirm with organisers
DWELL_BINS = [0, 15, 60, 180, float("inf")] # dwell segments (in dwell_time units)
DWELL_LABELS = ["pass-through (<15)", "short (15–60)", "medium (60–180)", "long (180+)"]
Z_THRESH = 3.0                              # anomaly threshold
SAMPLE_ROWS = 300_000                       # for raw-point plots
# ----------------------------------------

spark.conf.set("spark.sql.session.timeZone", LOCAL_TZ)

plt.rcParams.update({
    "figure.figsize": (13, 5), "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13,
})
sns.set_palette("tab10")

def show():
    plt.tight_layout()
    plt.show()

df_raw = spark.table(TABLE)
ts_expr = F.to_timestamp("timestamp") if dict(df_raw.dtypes)["timestamp"] == "string" else F.col("timestamp")

df = (df_raw
      .withColumn("ts", ts_expr)
      .withColumn("dwell_time", F.col("dwell_time").cast("double"))
      .withColumn("date", F.to_date("ts"))
      .withColumn("hour", F.hour("ts"))
      .withColumn("minute", F.minute("ts"))
      .withColumn("second", F.second("ts"))
      .withColumn("dow", F.expr("weekday(ts)"))               # 0 = Mon
      .withColumn("dow_name", F.date_format("ts", "EEE"))
      .withColumn("is_weekend", F.col("dow") >= 5)
      .withColumn("slot_of_day", F.col("hour") * 4 + F.floor(F.col("minute") / 15))
      .withColumn("slot15", F.expr("timestamp_seconds(floor(unix_timestamp(ts) / 900) * 900)"))
     ).cache()

N = df.count()
print(f"Rows: {N:,}")

# COMMAND ----------

# MAGIC %md ## 1. Data overview & quality

# COMMAND ----------

df_raw.printSchema()
display(df.limit(20))

# COMMAND ----------

# Nulls, distinct counts, duplicates
base_cols = df_raw.columns
nulls = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in base_cols]).toPandas().T
nulls.columns = ["nulls"]
nulls["null_pct"] = nulls["nulls"] / N * 100
nulls["approx_distinct"] = df.select([F.approx_count_distinct(c).alias(c) for c in base_cols]).toPandas().T[0]
display(nulls.reset_index().rename(columns={"index": "column"}))

dup = N - df_raw.dropDuplicates().count()
print(f"Exact duplicate rows: {dup:,} ({dup / N:.2%})")

# COMMAND ----------

display(df.select("dwell_time", "latitude", "longitude").summary(
    "count", "mean", "stddev", "min", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "max"))

# COMMAND ----------

# Time coverage
rng = df.agg(F.min("ts").alias("first"), F.max("ts").alias("last"),
             F.countDistinct("date").alias("n_days"), F.countDistinct("slot15").alias("n_slots")).first()
expected_slots = int((rng["last"] - rng["first"]).total_seconds() // 900) + 1
print(f"First: {rng['first']}  Last: {rng['last']}  Days: {rng['n_days']}")
print(f"15-min slots with data: {rng['n_slots']:,} / {expected_slots:,} expected ({rng['n_slots'] / expected_slots:.1%} coverage)")

# COMMAND ----------

# Coordinate consistency — each location should map to exactly one lat/lon
coord_check = (df.groupBy("location_name")
                 .agg(F.countDistinct("latitude", "longitude").alias("n_coord_pairs"),
                      F.min("latitude").alias("lat_min"), F.max("latitude").alias("lat_max"),
                      F.min("longitude").alias("lon_min"), F.max("longitude").alias("lon_max"))
                 .orderBy(F.desc("n_coord_pairs")))
display(coord_check)

shared = (df.select("location_name", "latitude", "longitude").distinct()
            .groupBy("latitude", "longitude").agg(F.collect_set("location_name").alias("names"),
                                                   F.count("*").alias("n"))
            .filter("n > 1"))
print("Coordinates shared by >1 location name:")
display(shared)

# COMMAND ----------

# Dwell-time sanity
dq = df.agg(
    F.sum((F.col("dwell_time") <= 0).cast("int")).alias("non_positive"),
    F.sum((F.col("dwell_time") != F.floor("dwell_time")).cast("int")).alias("non_integer"),
    F.max("dwell_time").alias("max"),
).first().asDict()
print(dq)

# Origin labels — spot typos / casing variants
display(df.groupBy("origin").count().orderBy("origin"))

# COMMAND ----------

# MAGIC %md ## 2. Synthetic-data artefact checks
# MAGIC Generated data often has tell-tale patterns (only certain minutes, uniform seconds, fixed daily shape). Knowing these stops us over-interpreting them.

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, col in zip(axes, ["minute", "second", "hour"]):
    p = df.groupBy(col).count().orderBy(col).toPandas()
    ax.bar(p[col], p["count"], color="steelblue")
    ax.set_title(f"Records by {col}")
    ax.set_xlabel(col)
axes[0].set_ylabel("records")
show()

# COMMAND ----------

# Records per location per day — is volume suspiciously constant?
lpd = df.groupBy("location_name", "date").count()
display(lpd.groupBy("location_name").agg(
    F.avg("count").alias("avg_daily"), F.stddev("count").alias("std_daily"),
    (F.stddev("count") / F.avg("count")).alias("coef_var")).orderBy("coef_var"))

# COMMAND ----------

# MAGIC %md ## 3. Volume by location & origin

# COMMAND ----------

loc_vol = df.groupBy("location_name").count().orderBy(F.desc("count")).toPandas()
loc_vol["share"] = loc_vol["count"] / loc_vol["count"].sum()
top_locs = loc_vol["location_name"].head(TOP_N).tolist()
print(f"Distinct locations: {len(loc_vol)}")

fig, ax = plt.subplots(figsize=(13, max(5, 0.35 * min(len(loc_vol), 40))))
p = loc_vol.head(40).iloc[::-1]
bars = ax.barh(p["location_name"], p["count"], color="steelblue")
for b, s in zip(bars, p["share"]):
    ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {s:.1%}", va="center", fontsize=9)
ax.set_title("Records by location (top 40)")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
show()

# COMMAND ----------

# Pareto — how concentrated is traffic?
cum = loc_vol["share"].cumsum().values
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.arange(1, len(cum) + 1), cum, marker="o")
ax.axhline(0.8, ls="--", color="grey")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlabel("number of locations (ranked by volume)")
ax.set_ylabel("cumulative share of records")
ax.set_title(f"Traffic concentration — {np.argmax(cum >= 0.8) + 1} locations carry 80% of records")
show()

# COMMAND ----------

org_vol = df.groupBy("origin").count().orderBy(F.desc("count")).toPandas()
org_vol["share"] = org_vol["count"] / org_vol["count"].sum()

fig, ax = plt.subplots(figsize=(13, max(5, 0.35 * min(len(org_vol), 40))))
p = org_vol.head(40).iloc[::-1]
bars = ax.barh(p["origin"], p["count"], color="darkorange")
for b, s in zip(bars, p["share"]):
    ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {s:.1%}", va="center", fontsize=9)
ax.set_title("Records by origin (top 40)")
show()

# COMMAND ----------

# MAGIC %md ## 4. Dwell-time distribution

# COMMAND ----------

qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
qv = df.approxQuantile("dwell_time", qs, 0.001)
dwell_q = dict(zip(qs, qv))
display(pd.DataFrame({"quantile": qs, "dwell_time": qv}))
P99 = dwell_q[0.99]
CAP = dwell_q[0.999]

# COMMAND ----------

BIN = max(1, round(CAP / 80))
hist = (df.filter(F.col("dwell_time") <= CAP)
          .withColumn("bin", F.floor(F.col("dwell_time") / BIN) * BIN)
          .groupBy("bin").count().orderBy("bin").toPandas())

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
axes[0].bar(hist["bin"], hist["count"], width=BIN * 0.9, color="teal")
axes[0].set_title(f"Dwell-time histogram (≤ p99.9 = {CAP:.0f}, bin = {BIN})")
axes[1].bar(hist["bin"], hist["count"], width=BIN * 0.9, color="teal")
axes[1].set_yscale("log")
axes[1].set_title("Same, log y-axis (tail shape)")
for a in axes:
    a.axvline(dwell_q[0.5], color="red", ls="--", label=f"median {dwell_q[0.5]:.0f}")
    a.axvline(P99, color="black", ls=":", label=f"p99 {P99:.0f}")
    a.set_xlabel("dwell_time")
    a.legend()
show()

# COMMAND ----------

# ECDF from 200 quantiles (exact-enough, no sampling)
grid = np.linspace(0, 1, 201)
ecdf_vals = df.approxQuantile("dwell_time", grid.tolist(), 0.001)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ecdf_vals, grid)
ax.set_xscale("log")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlabel("dwell_time (log)")
ax.set_ylabel("share of records ≤ x")
ax.set_title("Dwell-time ECDF")
show()

# COMMAND ----------

# Box plots per location from precomputed quantiles (all rows, no sampling)
bq = (df.filter(F.col("location_name").isin(top_locs))
        .groupBy("location_name")
        .agg(F.expr("percentile_approx(dwell_time, array(0.05, 0.25, 0.5, 0.75, 0.95), 1000)").alias("q"),
             F.avg("dwell_time").alias("mean"))
        .toPandas())
bq["median"] = bq["q"].apply(lambda q: q[2])
bq = bq.sort_values("median")

fig = go.Figure()
for _, r in bq.iterrows():
    fig.add_trace(go.Box(x=[r["location_name"]], lowerfence=[r["q"][0]], q1=[r["q"][1]], median=[r["q"][2]],
                         q3=[r["q"][3]], upperfence=[r["q"][4]], mean=[r["mean"]],
                         name=r["location_name"], showlegend=False))
fig.update_layout(title="Dwell time by location (whiskers = p5–p95, dashed = mean)",
                  yaxis_title="dwell_time", height=550)
fig.show()

# COMMAND ----------

# Hexbin: time of day vs dwell (sample)
frac = min(1.0, SAMPLE_ROWS / N)
sm = (df.sample(fraction=frac, seed=42)
        .select((F.col("hour") + F.col("minute") / 60).alias("hour_frac"), "dwell_time")
        .filter(F.col("dwell_time") <= CAP).toPandas())
fig, ax = plt.subplots(figsize=(13, 5))
hb = ax.hexbin(sm["hour_frac"], sm["dwell_time"], gridsize=60, cmap="viridis", bins="log", mincnt=1)
plt.colorbar(hb, label="log count")
ax.set_xlabel("time of day (hour)")
ax.set_ylabel("dwell_time")
ax.set_title(f"Arrival time vs dwell (sample of {len(sm):,})")
show()

# COMMAND ----------

# MAGIC %md ## 5. Temporal patterns

# COMMAND ----------

daily = (df.groupBy("date").agg(F.count("*").alias("records"),
                               F.expr("percentile_approx(dwell_time, 0.5)").alias("median_dwell"))
           .orderBy("date").toPandas())
daily["date"] = pd.to_datetime(daily["date"])
daily["roll7"] = daily["records"].rolling(7, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(daily["date"], daily["records"], alpha=0.5, marker=".", label="daily")
ax.plot(daily["date"], daily["roll7"], lw=2.5, label="7-day avg")
for d in daily.loc[daily["date"].dt.dayofweek >= 5, "date"]:
    ax.axvspan(d, d + pd.Timedelta(days=1), color="grey", alpha=0.12)
ax.set_title("Daily records (grey = weekend)")
ax.legend()
show()

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
h = df.groupBy("hour").count().orderBy("hour").toPandas()
axes[0].bar(h["hour"], h["count"] / rng["n_days"], color="steelblue")
axes[0].set_title("Average records per hour of day")
axes[0].set_xticks(range(24))

dw = (df.groupBy("date", "dow", "dow_name").count()
        .groupBy("dow", "dow_name").agg(F.avg("count").alias("avg")).orderBy("dow").toPandas())
axes[1].bar(dw["dow_name"], dw["avg"], color=["steelblue"] * 5 + ["darkorange"] * 2)
axes[1].set_title("Average records per day of week")
show()

# COMMAND ----------

# Day-of-week × hour heatmap (average per calendar day)
dh = (df.groupBy("date", "dow", "dow_name", "hour").count()
        .groupBy("dow", "dow_name", "hour").agg(F.avg("count").alias("avg")).toPandas())
piv = dh.pivot_table(index=["dow", "dow_name"], columns="hour", values="avg").sort_index()
piv.index = piv.index.get_level_values(1)
plt.figure(figsize=(16, 4.5))
sns.heatmap(piv, cmap="rocket_r", cbar_kws={"label": "avg records"})
plt.title("Average records: day of week × hour")
show()

# COMMAND ----------

# Weekday vs weekend hourly profile
wk = (df.groupBy("is_weekend", "date", "hour").count()
        .groupBy("is_weekend", "hour").agg(F.avg("count").alias("avg")).orderBy("hour").toPandas())
fig, ax = plt.subplots(figsize=(12, 5))
for flag, lab in [(False, "weekday"), (True, "weekend")]:
    s = wk[wk["is_weekend"] == flag]
    ax.plot(s["hour"], s["avg"], marker="o", label=lab)
ax.set_xticks(range(24))
ax.set_title("Hourly profile: weekday vs weekend")
ax.legend()
show()

# COMMAND ----------

# Location × hour — share of each location's daily volume (commuter vs nightlife signatures)
lh = df.groupBy("location_name", "hour").count().toPandas()
lh_piv = lh.pivot(index="location_name", columns="hour", values="count").fillna(0)
lh_share = lh_piv.div(lh_piv.sum(axis=1), axis=0)

plt.figure(figsize=(16, max(5, 0.4 * len(top_locs))))
sns.heatmap(lh_share.loc[top_locs], cmap="mako_r", cbar_kws={"label": "share of location's volume"})
plt.title("When is each location busy? (row-normalised)")
show()

# COMMAND ----------

fig, ax = plt.subplots(figsize=(14, 6))
for loc in top_locs[:8]:
    ax.plot(lh_share.columns, lh_share.loc[loc], marker=".", label=loc)
ax.set_xticks(range(24))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Hourly profile — top 8 locations")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
show()

# COMMAND ----------

# 15-minute time series (interactive, zoomable)
ts15 = (df.filter(F.col("location_name").isin(top_locs[:6]))
          .groupBy("location_name", "slot15").count().orderBy("slot15").toPandas())
fig = px.line(ts15, x="slot15", y="count", color="location_name",
              title="Records per 15 minutes — top 6 locations")
fig.update_xaxes(rangeslider_visible=True)
fig.show()

# COMMAND ----------

# MAGIC %md ## 6. Spatial patterns

# COMMAND ----------

loc_geo = (df.groupBy("location_name")
             .agg(F.first("latitude").alias("lat"), F.first("longitude").alias("lon"),
                  F.count("*").alias("records"),
                  F.expr("percentile_approx(dwell_time, 0.5)").alias("median_dwell"),
                  F.countDistinct("origin").alias("n_origins"))
             .toPandas())

fig = px.scatter_mapbox(loc_geo, lat="lat", lon="lon", size="records", color="median_dwell",
                        hover_name="location_name", hover_data=["records", "median_dwell", "n_origins"],
                        size_max=45, zoom=10.5, height=650, color_continuous_scale="Turbo",
                        mapbox_style="open-street-map",
                        title="Locations — size = volume, colour = median dwell")
fig.show()

# COMMAND ----------

fig = px.density_mapbox(loc_geo, lat="lat", lon="lon", z="records", radius=35, zoom=10.5, height=650,
                        mapbox_style="open-street-map", title="Traffic density")
fig.show()

# COMMAND ----------

# Animated: how activity moves across the city through the day
geo_hour = (df.groupBy("location_name", "hour")
              .agg(F.count("*").alias("records"), F.first("latitude").alias("lat"),
                   F.first("longitude").alias("lon"))
              .toPandas())
# fill missing location-hour combos so points don't vanish between frames
full = pd.MultiIndex.from_product([loc_geo["location_name"], range(24)], names=["location_name", "hour"])
geo_hour = (geo_hour.set_index(["location_name", "hour"]).reindex(full).reset_index()
            .drop(columns=["lat", "lon"])
            .merge(loc_geo[["location_name", "lat", "lon"]], on="location_name"))
geo_hour["records"] = geo_hour["records"].fillna(0) / rng["n_days"]
geo_hour = geo_hour.sort_values("hour")

fig = px.scatter_mapbox(geo_hour, lat="lat", lon="lon", size="records", color="records",
                        animation_frame="hour", hover_name="location_name", size_max=45, zoom=10.5,
                        height=650, range_color=[0, geo_hour["records"].max()],
                        color_continuous_scale="Inferno", mapbox_style="open-street-map",
                        title="Average records by hour of day")
fig.show()

# COMMAND ----------

# MAGIC %md ## 7. Origin → destination flows

# COMMAND ----------

od = df.groupBy("origin", "location_name").count().toPandas()
od_piv = od.pivot(index="origin", columns="location_name", values="count").fillna(0)
od_piv = od_piv.loc[od_piv.sum(axis=1).sort_values(ascending=False).index,
                    od_piv.sum(axis=0).sort_values(ascending=False).index]

fig, axes = plt.subplots(1, 2, figsize=(24, max(6, 0.35 * len(od_piv))))
sns.heatmap(od_piv.div(od_piv.sum(axis=1), axis=0), cmap="Blues", ax=axes[0],
            cbar_kws={"label": "share of origin's trips"})
axes[0].set_title("Where does each origin go? (row-normalised)")
sns.heatmap(od_piv.div(od_piv.sum(axis=0), axis=1), cmap="Oranges", ax=axes[1],
            cbar_kws={"label": "share of destination's visitors"})
axes[1].set_title("Who comes to each location? (column-normalised)")
show()

# COMMAND ----------

top_od = od.nlargest(25, "count").copy()
top_od["pair"] = top_od["origin"] + " → " + top_od["location_name"]
fig, ax = plt.subplots(figsize=(13, 8))
ax.barh(top_od["pair"][::-1], top_od["count"][::-1], color="slateblue")
ax.set_title("Top 25 origin → destination pairs")
show()

# COMMAND ----------

# Sankey of the 40 biggest flows
sk = od.nlargest(40, "count")
origins = sk["origin"].unique().tolist()
dests = sk["location_name"].unique().tolist()
fig = go.Figure(go.Sankey(
    node=dict(label=origins + dests, pad=12, thickness=16,
              color=["#4C78A8"] * len(origins) + ["#F58518"] * len(dests)),
    link=dict(source=[origins.index(o) for o in sk["origin"]],
              target=[len(origins) + dests.index(d) for d in sk["location_name"]],
              value=sk["count"].tolist())))
fig.update_layout(title="Top 40 origin → destination flows", height=750)
fig.show()

# COMMAND ----------

# Catchment breadth: how many origins feed each location, and how evenly (Shannon entropy)
w_loc = Window.partitionBy("location_name")
catch = (df.groupBy("location_name", "origin").count()
           .withColumn("p", F.col("count") / F.sum("count").over(w_loc))
           .groupBy("location_name")
           .agg((-F.sum(F.col("p") * F.log("p"))).alias("origin_entropy"),
                F.count("*").alias("n_origins"),
                F.max("p").alias("top_origin_share"))
           .toPandas()
           .merge(loc_vol[["location_name", "count"]], on="location_name"))

fig, ax = plt.subplots(figsize=(12, 6))
sc = ax.scatter(catch["origin_entropy"], catch["top_origin_share"], s=catch["count"] / catch["count"].max() * 1500,
                c=catch["n_origins"], cmap="viridis", alpha=0.7, edgecolor="k")
for _, r in catch.iterrows():
    ax.annotate(r["location_name"], (r["origin_entropy"], r["top_origin_share"]), fontsize=8)
plt.colorbar(sc, label="# origins")
ax.set_xlabel("origin entropy (higher = draws from many neighbourhoods)")
ax.set_ylabel("share from single biggest origin")
ax.set_title("Catchment: local vs city-wide destinations (size = volume)")
show()

# COMMAND ----------

# Hourly shape of the top 6 OD pairs
top6 = od.nlargest(6, "count")[["origin", "location_name"]]
odh = (df.join(spark.createDataFrame(top6), ["origin", "location_name"])
         .groupBy("origin", "location_name", "hour").count().toPandas())
odh["pair"] = odh["origin"] + " → " + odh["location_name"]
fig = px.line(odh.sort_values("hour"), x="hour", y="count", color="pair", markers=True,
              title="Hourly profile of the top 6 OD pairs")
fig.show()

# COMMAND ----------

# MAGIC %md ## 8. Dwell behaviour by place & time

# COMMAND ----------

loc_dwell = (df.groupBy("location_name")
               .agg(F.count("*").alias("records"), F.avg("dwell_time").alias("mean_dwell"),
                    F.expr("percentile_approx(dwell_time, 0.5)").alias("median_dwell"),
                    F.expr("percentile_approx(dwell_time, 0.9)").alias("p90_dwell"))
               .toPandas().sort_values("median_dwell"))

fig, ax = plt.subplots(figsize=(13, max(5, 0.35 * len(loc_dwell))))
y = np.arange(len(loc_dwell))
ax.barh(y - 0.2, loc_dwell["median_dwell"], height=0.4, label="median")
ax.barh(y + 0.2, loc_dwell["p90_dwell"], height=0.4, label="p90", alpha=0.7)
ax.set_yticks(y)
ax.set_yticklabels(loc_dwell["location_name"])
ax.set_title("Dwell time by location")
ax.legend()
show()

# COMMAND ----------

# Quadrant: busy + short dwell = transit hubs; busy + long dwell = gathering places (crowd-safety focus)
fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(loc_dwell["records"], loc_dwell["median_dwell"], s=120, alpha=0.7, edgecolor="k")
for _, r in loc_dwell.iterrows():
    ax.annotate(r["location_name"], (r["records"], r["median_dwell"]), fontsize=8, xytext=(4, 4),
                textcoords="offset points")
ax.axvline(loc_dwell["records"].median(), ls="--", color="grey")
ax.axhline(loc_dwell["median_dwell"].median(), ls="--", color="grey")
ax.set_xscale("log")
ax.set_xlabel("records (log)")
ax.set_ylabel("median dwell")
ax.set_title("Volume vs dwell — transit hubs (bottom-right) vs gathering places (top-right)")
show()

# COMMAND ----------

hd = (df.groupBy("hour")
        .agg(F.expr("percentile_approx(dwell_time, array(0.25, 0.5, 0.75, 0.9), 1000)").alias("q"))
        .orderBy("hour").toPandas())
q = np.vstack(hd["q"].values)
fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(hd["hour"], q[:, 0], q[:, 2], alpha=0.25, label="IQR")
ax.plot(hd["hour"], q[:, 1], marker="o", label="median")
ax.plot(hd["hour"], q[:, 3], ls="--", label="p90")
ax.set_xticks(range(24))
ax.set_title("Dwell time by hour of arrival")
ax.legend()
show()

# COMMAND ----------

ldh = (df.filter(F.col("location_name").isin(top_locs))
         .groupBy("location_name", "hour")
         .agg(F.expr("percentile_approx(dwell_time, 0.5)").alias("median_dwell")).toPandas()
         .pivot(index="location_name", columns="hour", values="median_dwell"))
plt.figure(figsize=(16, max(5, 0.4 * len(top_locs))))
sns.heatmap(ldh.loc[top_locs], cmap="YlGnBu", cbar_kws={"label": "median dwell"})
plt.title("Median dwell: location × hour")
show()

# COMMAND ----------

od_dw = (df.groupBy("origin").agg(F.expr("percentile_approx(dwell_time, 0.5)").alias("median_dwell"),
                                  F.count("*").alias("records"))
           .toPandas().sort_values("median_dwell"))
fig, ax = plt.subplots(figsize=(12, max(5, 0.3 * len(od_dw))))
ax.barh(od_dw["origin"], od_dw["median_dwell"], color="darkorange")
ax.set_title("Median dwell by origin")
show()

# COMMAND ----------

# Dwell segments per location (stacked share)
seg = F.when(F.col("dwell_time") < DWELL_BINS[1], DWELL_LABELS[0])
for lo, lab in zip(DWELL_BINS[2:-1], DWELL_LABELS[1:-1]):
    seg = seg.when(F.col("dwell_time") < lo, lab)
seg = seg.otherwise(DWELL_LABELS[-1])

segs = (df.withColumn("segment", seg).groupBy("location_name", "segment").count().toPandas()
          .pivot(index="location_name", columns="segment", values="count").fillna(0)[DWELL_LABELS])
segs = segs.div(segs.sum(axis=1), axis=0).sort_values(DWELL_LABELS[0])
ax = segs.plot(kind="barh", stacked=True, figsize=(13, max(5, 0.35 * len(segs))), colormap="viridis")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Dwell segments by location")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
show()

# COMMAND ----------

# MAGIC %md ## 9. Crowd occupancy — devices present per 15 minutes
# MAGIC Each record is present from `timestamp` to `timestamp + dwell_time`. Exploding into the 15-min slots it covers gives an estimate of how many people are at a location at once — the core crowd-safety metric. Check `DWELL_UNIT_SECONDS` first.

# COMMAND ----------

occ = (df.select("location_name",
                 F.unix_timestamp("ts").alias("start_s"),
                 (F.unix_timestamp("ts") + F.col("dwell_time") * DWELL_UNIT_SECONDS).cast("long").alias("end_s"))
         .withColumn("slot_s", F.explode(F.sequence(F.floor(F.col("start_s") / 900) * 900,
                                                    F.floor(F.col("end_s") / 900) * 900,
                                                    F.lit(900).cast("long"))))
         .withColumn("slot", F.expr("timestamp_seconds(slot_s)"))
         .groupBy("location_name", "slot").agg(F.count("*").alias("present")))
occ.cache()

occ_stats = (occ.groupBy("location_name")
               .agg(F.max("present").alias("peak_present"),
                    F.expr("percentile_approx(present, 0.5)").alias("median_present"),
                    F.expr("percentile_approx(present, 0.99)").alias("p99_present"))
               .toPandas().sort_values("peak_present"))

fig, ax = plt.subplots(figsize=(13, max(5, 0.35 * len(occ_stats))))
y = np.arange(len(occ_stats))
ax.barh(y, occ_stats["peak_present"], label="peak", alpha=0.6)
ax.barh(y, occ_stats["p99_present"], label="p99", alpha=0.8)
ax.barh(y, occ_stats["median_present"], label="median")
ax.set_yticks(y)
ax.set_yticklabels(occ_stats["location_name"])
ax.set_title("Estimated devices present per 15 min")
ax.legend()
show()

# COMMAND ----------

occ_top = occ.filter(F.col("location_name").isin(top_locs[:6])).orderBy("slot").toPandas()
fig = px.line(occ_top, x="slot", y="present", color="location_name",
              title="Estimated devices present — top 6 locations")
fig.update_xaxes(rangeslider_visible=True)
fig.show()

# COMMAND ----------

# MAGIC %md ## 10. Security / anomaly signals

# COMMAND ----------

# Arrival surges: z-score of each 15-min count vs same location, same weekday/weekend, same time-of-day slot
c15 = df.groupBy("location_name", "is_weekend", "slot_of_day", "slot15").count()
base = c15.groupBy("location_name", "is_weekend", "slot_of_day").agg(
    F.avg("count").alias("mu"), F.stddev("count").alias("sd"), F.count("*").alias("n_obs"))
anom = (c15.join(base, ["location_name", "is_weekend", "slot_of_day"])
           .filter((F.col("n_obs") >= 3) & (F.col("sd") > 0))
           .withColumn("z", (F.col("count") - F.col("mu")) / F.col("sd")))
anom.cache()

print(f"Slots with |z| > {Z_THRESH}: {anom.filter(F.abs('z') > Z_THRESH).count():,} of {anom.count():,}")
display(anom.orderBy(F.desc("z")).select("location_name", "slot15", "count", "mu", "sd", "z").limit(25))

# COMMAND ----------

az = (anom.filter(F.col("z") > Z_THRESH).groupBy("location_name").count()
          .orderBy(F.desc("count")).toPandas())
fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(az))))
ax.barh(az["location_name"][::-1], az["count"][::-1], color="crimson")
ax.set_title(f"Surge slots (z > {Z_THRESH}) by location")
show()

# COMMAND ----------

if len(az):
    worst = az["location_name"].iloc[0]
    s = anom.filter(F.col("location_name") == worst).orderBy("slot15").toPandas()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["slot15"], y=s["count"], mode="lines", name="records / 15 min"))
    fig.add_trace(go.Scatter(x=s["slot15"], y=s["mu"], mode="lines", name="expected", line=dict(dash="dash")))
    hi = s[s["z"] > Z_THRESH]
    fig.add_trace(go.Scatter(x=hi["slot15"], y=hi["count"], mode="markers", name="surge",
                             marker=dict(color="red", size=9)))
    fig.update_layout(title=f"Surges at {worst}", height=500)
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()

# COMMAND ----------

# Surge ratio: how spiky is each location? (p99 slot vs median slot)
spk = (c15.groupBy("location_name")
          .agg(F.expr("percentile_approx(count, 0.99)").alias("p99"),
               F.expr("percentile_approx(count, 0.5)").alias("p50"))
          .withColumn("spike_ratio", F.col("p99") / F.col("p50"))
          .toPandas().sort_values("spike_ratio"))
fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(spk))))
ax.barh(spk["location_name"], spk["spike_ratio"], color="indianred")
ax.set_title("Spikiness: p99 ÷ median arrivals per 15 min")
show()

# COMMAND ----------

# Late-night activity (00:00–04:59) and long-dwell (> global p99) — loitering-type signals
night = (df.groupBy("location_name")
           .agg(F.avg((F.col("hour") < 5).cast("int")).alias("night_share"),
                F.avg((F.col("dwell_time") > P99).cast("int")).alias("long_dwell_share"))
           .toPandas())

fig, axes = plt.subplots(1, 2, figsize=(20, max(5, 0.35 * len(night))))
for ax, col, color, title in [(axes[0], "night_share", "midnightblue", "Share of records 00:00–04:59"),
                              (axes[1], "long_dwell_share", "darkred", f"Share of records with dwell > p99 ({P99:.0f})")]:
    p = night.sort_values(col)
    ax.barh(p["location_name"], p[col], color=color)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(title)
show()

# COMMAND ----------

ld = (df.filter(F.col("dwell_time") > P99).groupBy("location_name", "hour").count().toPandas()
        .pivot(index="location_name", columns="hour", values="count").fillna(0))
plt.figure(figsize=(16, max(5, 0.4 * len(ld))))
sns.heatmap(ld, cmap="Reds", cbar_kws={"label": "long-dwell records"})
plt.title(f"Where and when do long dwells (> {P99:.0f}) happen?")
show()

# COMMAND ----------

# MAGIC %md ## 11. Correlation & similarity

# COMMAND ----------

# Locations with similar daily rhythms (hourly profile correlation) — candidate clusters
prof_corr = lh_share.T.corr()
g = sns.clustermap(prof_corr, cmap="vlag", center=0, figsize=(12, 12))
g.fig.suptitle("Hourly-profile correlation between locations", y=1.02)
plt.show()

# COMMAND ----------

# Do locations rise and fall together day to day?
if rng["n_days"] >= 5:
    dl = lpd.toPandas().pivot(index="date", columns="location_name", values="count").fillna(0)
    g = sns.clustermap(dl.corr(), cmap="vlag", center=0, figsize=(12, 12))
    g.fig.suptitle("Daily-volume correlation between locations", y=1.02)
    plt.show()

# COMMAND ----------

num = df.sample(fraction=frac, seed=7).select("latitude", "longitude", "dwell_time", "hour", "dow",
                                             F.col("is_weekend").cast("int").alias("is_weekend")).toPandas()
plt.figure(figsize=(7, 6))
sns.heatmap(num.corr(method="spearman"), annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Spearman correlation (sample)")
show()

# COMMAND ----------

# MAGIC %md ## 12. Location KPI table
# MAGIC One row per location — feeds the solution design (hub classification, alert thresholds, dashboard).

# COMMAND ----------

peak = lh_piv.idxmax(axis=1).rename("peak_hour")
peak_ratio = (lh_piv.max(axis=1) / lh_piv.mean(axis=1)).rename("peak_to_avg_hour")

kpi = (loc_geo[["location_name", "lat", "lon", "records"]]
       .merge(loc_dwell[["location_name", "mean_dwell", "median_dwell", "p90_dwell"]], on="location_name")
       .merge(catch[["location_name", "n_origins", "origin_entropy", "top_origin_share"]], on="location_name")
       .merge(night, on="location_name")
       .merge(spk[["location_name", "spike_ratio"]], on="location_name", how="left")
       .merge(occ_stats, on="location_name", how="left")
       .merge(az.rename(columns={"count": "surge_slots"}), on="location_name", how="left")
       .merge(peak.reset_index(), on="location_name")
       .merge(peak_ratio.reset_index(), on="location_name"))
kpi["surge_slots"] = kpi["surge_slots"].fillna(0).astype(int)
kpi["share_of_records"] = kpi["records"] / kpi["records"].sum()
kpi = kpi.sort_values("records", ascending=False)
display(kpi)

spark.createDataFrame(kpi).write.mode("overwrite").option("overwriteSchema", "true") \
     .saveAsTable(f"{OUT_SCHEMA}.location_kpis")
