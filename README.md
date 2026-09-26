# Smart Community Solution — Rogers × Databricks

**Community Pulse:** a planning and safety tool built on Rogers cell-tower data. For three very different places in Metro Vancouver (a downtown transit hub, a university campus and a shopping mall), it shows how many people are there, when they arrive and leave, where they live, and when something is unusual.

---

## 1. Data

Three synthetic datasets with the same schema, in `rogersdatabricks.default`:

| Table | Site | Type of place |
|---|---|---|
| `synthetic_data_waterfront` | Waterfront Station | Downtown transit hub, business and tourism district |
| `synthetic_data_ubc` | UBC | University campus |
| `synthetic_data_royal_mall` | Park Royal Mall | Regional shopping centre, West Vancouver |

| Column | Meaning |
|---|---|
| `location_name` | Name of the point of interest |
| `longitude`, `latitude` | Coordinates of the point of interest |
| `timestamp` | Time the device attached to the cell tower |
| `origin` | **Home location** of the device |
| `dwell_time` | **Minutes** the device stayed attached to the cell tower |

### What the data can and cannot tell us

| It tells us | It does not tell us |
|---|---|
| How many devices are around the site, and when they arrive and leave | Whether someone was in the station, an office or on the street. The tower's coverage is larger than the named place. |
| How long each device stayed | How people travelled (transit, car, walking) or why they came |
| Each device's home area, resident vs visitor | Unique people: there is no device ID, so one person can produce several records |
| | Movement between sites: we can't follow a device from one tower to another |

**Wording rule:** say "device connections", "devices present" or "home area". Never say "riders", "unique people" or "trips".

### Waterfront findings so far (`rogers_waterfront_eda-2.ipynb`)

- 8,058,918 records after removing 1,094 duplicates. Nov 1 2025 to Aug 31 2026, 304 days, no gaps.
- About 26.5K connections per day. Weekday peaks are at 08:00 and 16:00–17:00 (the biggest is Wednesday at 16:00). Weekends are flatter and centred on midday.
- About 14 of the 36 home areas supply 80% of records.
- About 30% of records are from non-local visitors (Ontario 16.7%, BC Other 7.7%, Alberta 2.7%, …).
- Port Moody, Pitt Meadows and Maple Ridge arrive in sharp spikes at 08:00 and 16:00–17:00, consistent with West Coast Express commuting.
- Dwell percentiles (minutes): p5 11, median 54, p90 166, p99 410, max 17,941 (invalid).
- For arrivals between 03:00 and 04:00, p90 dwell is about 480 minutes, against about 165 during the day.

### Known data issues

- **Invalid dwell:** p99.9 equals the max (17,941 minutes, about 12.5 days), so there is a block of impossible values. `rogers_combined_eda.ipynb` flags anything over 24h and excludes it from dwell stats and occupancy.
- **Generator artefact:** at Waterfront, minutes :30–:59 carry about 7% more records than :00–:29. The notebook checks this per site. Don't present it as a finding.
- **Daily-resolution bug in `rogers_waterfront_eda-2.ipynb`:** with 304 days, sections 7–9 (occupancy, surges) ran on daily buckets. Superseded by the 15-minute analysis in `rogers_combined_eda.ipynb`.

---

## 2. What we're proposing

### The problem
Cities, transit agencies and site operators plan staffing, service and security from fare taps, ticket sales and gut feel. None of these shows **everyone** who is at a place: residents and visitors, riders and non-riders, how long they stay and where they live. Cell-tower data does, and it does so without identifying anyone.

### The product: Community Pulse
A Databricks-based data product that Rogers can offer site by site. It turns past tower connections into three things planners can act on:

1. **Rhythm and forecast (planning)**
   - "Typical" weekday, weekend and holiday profiles for each site at 15-minute resolution.
   - A 30–90 day forecast of arrivals and devices present.
   - A resident vs visitor split and where visitors come from.
2. **Crowd safety (security)**
   - How many devices are present per 15 minutes, compared with the normal level for that weekday and time slot.
   - Flags for unusual surges, crowds that are too large, and sudden drops (possible disruptions).
   - A view of long overnight stays.
   - Every flag comes with a plain-English reason.
3. **Travel demand (transit)**
   - Departures per 15 minutes, grouped by the home corridor people are heading back to.
   - Compared with scheduled outbound service from TransLink's GTFS timetables.
   - Shows when many people need to travel somewhere and service is thin.
   - All three sites are bus/transit exchanges: Waterfront, UBC Exchange and Park Royal.

**Why three sites matter:** the same pipeline serves three kinds of place (a hub, a campus and a mall) and therefore three kinds of client. That's the scaling story: add a tower, get a new client.

### Who it's for

| Stakeholder | What they get |
|---|---|
| **TransLink** | Outbound demand by corridor against scheduled service at three exchanges; evidence for service changes |
| **City of Vancouver / West Vancouver, Transit Police, UBC Campus Security** | Crowd baselines, anomaly flags, patrol timing, event and holiday planning |
| **Rogers** | A repeatable B2B data product, shared with clients through Delta Sharing, privacy-safe by design |
| **PwC** | A business case by client type, plus a privacy and governance framework (PIPEDA, BC FIPPA) |
| **Databricks** | End to end on the platform: Lakeflow pipeline, Unity Catalog, MLflow, AI/BI dashboards, Genie, Delta Sharing |

### Using only past data
We don't need live data to prove value:
- **Backtest:** train on Nov–Jun, predict Jul–Aug, and show the anomaly logic would have flagged the unusual days in advance. The baselines in `rogers_combined_eda.ipynb` only use past weeks, so the same code works on a live feed.
- **Forward plan:** forecast Sep–Dec 2026 and produce a concrete staffing and patrol plan.
- **Replay:** stream a past day through the pipeline to show what the live version would look like.

### Outside data to add (all available for the past)
- **BC statutory holidays:** hard-coded in the notebook.
- **Weather:** Open-Meteo historical API, hourly.
- **TransLink GTFS:** scheduled departures at each exchange.
- **Event and seasonal calendars:** cruise ship schedules at Canada Place, Convention Centre, Rogers Arena and BC Place events, UBC term dates.
- **City of Vancouver open data and Census:** local-area populations for visits per 1,000 residents.
- **VPD crime open data:** incidents per 1,000 devices present, i.e. a risk rate rather than a raw count.

Before claiming any relationship, test for it: the data is synthetic, so it may not follow real events.

### Privacy and governance
- Aggregates only. No device-level data leaves the silver layer.
- Suppress small counts (e.g. under 10) in shared tables.
- Access is granted per client in Unity Catalog.

---

## 3. Pipeline

```
rogersdatabricks.default.synthetic_data_{waterfront, ubc, royal_mall}
   ▼
silver.silver_visits  cleaned, deduped, dwell_valid flag, 15-min slots, resident/visitor, dwell segments
   ▼                  (rogersdatabricks.silver)
gold.* (rogersdatabricks.gold)
gold_flow_15m         site × 15-min: arrivals, departures, occupancy, baselines, z-scores, flags
gold_site_origin      site × home area: volume, shares, dwell, night share
gold_site_day         site × day: volume, peak occupancy, anomaly counts, holiday
gold_mix_hourly       site × hour: home-area-mix divergence and the home area driving it
gold_site_kpis        site: headline numbers
   ▼
AI/BI dashboard · Genie · forecast model (MLflow) · Delta Sharing
```

## 4. Build plan

| # | Step | Status |
|---|---|---|
| 1 | Waterfront EDA | Done (daily-resolution sections superseded) |
| 2 | Combined three-site EDA at 15 minutes, plus silver/gold tables (`rogers_combined_eda.ipynb`) | Written, needs to be run on Databricks |
| 3 | Decide the story from the three-site comparison | Next |
| 4 | Add external data: holidays, weather, GTFS, then events | |
| 5 | Forecast model and backtest (MLflow) | |
| 6 | Dashboard (Rhythm, Crowd safety, Travel demand) and Genie | |
| 7 | Business case, governance and demo script | |

## 5. Repo

| File | What it is |
|---|---|
| `rogers_combined_eda.ipynb` | Combined EDA of all three sites at 15 minutes; builds `silver_visits` and the gold tables |
| `rogers_waterfront_eda-2.ipynb` | Waterfront EDA (latest run, with outputs) |
| `rogers_waterfront_eda.ipynb` | Earlier Waterfront EDA |
