#!/usr/bin/env python3
# %% [markdown]
# # Escooter-Trip Simulation Notebook
# This notebook shows a simple pipeline to generate a large number of escooter trips in a German cities
# using OSMnx and NetworkX. Each cell is prefixed with `%%` so you can copy-paste directly into a Jupyter notebook.
# The trips are modeled using the following steps:
# 1. start point and start time of every ride is chosen randomly
# 2. The route traverses through nodes in a direction of a random bearing, choosing nodes with the smallest bearing difference
# 3. The speed between the nodes along the trip is modeled with a uniform distribution between 10-20km/h (as 20 is max legal speed in germany)
# 4. Using the speed and the distance between the nodes, the time to travel between the nodes is computed
# 5. Using the route, timestart, and time to travel between nodes of the route, the single trips representing the position and time are generated
#   * those trips are representing the shared escooter reporting during trip
#   * those trips are exported in CSV format to "escooter_trips.csv"

# %%
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from networkx import MultiDiGraph
from scipy.stats import lognorm, beta
from typing import cast, Any, List, Optional, Tuple
import logging
import numpy as np
import osmnx as ox
import pandas as pd
import random
import time
import uuid


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Set the level, this allows to change the level without restarting the interpreter
logging.getLogger().setLevel(logging.INFO)

seed = 14
random.seed(seed)
np.random.seed(seed)
logging.info(f"Using seed {seed} for random number generation")

# optionally enable logging for progress
# ox.config(log_console=True, use_cache=True)

# %%
# 1. Load the bike network graph for Berlin (or your city of choice)
city = "Berlin, Germany"
logging.info(f"Loading bike network graph for {city}")
G = ox.graph_from_place(city, network_type="bike")
logging.info("Graph loaded, projecting to latlong")
G = ox.project_graph(G, to_latlong=True)
logging.info(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

# Add speed_kph attributes to edges, modelling the speed of escooters on them
# Model the speed using a beta distribution
logging.info("Adding speed_kph attribute to edges")
for u, v, k, data in G.edges(keys=True, data=True):
    # Parameters of the beta distribution
    alpha = 8
    beta_param = 1
    scooter_speed = float(beta.rvs(alpha, beta_param) * 20)  # scale to 0-20 km/h
    data["speed_kph"] = scooter_speed
logging.info("Speed_kph attribute added to edges")

logging.info("Adding travel times to edges")
G = ox.add_edge_travel_times(G)

logging.info("Adding edge bearings to edges")
G = ox.add_edge_bearings(G)


# %%
# Generate synthetic ride lengths (meters)
def get_random_route_length() -> np.float64:
    return np.round(
        lognorm.rvs(
            0.5,  # σ of the underlying normal
            loc=0,  # loc should be 0 for pure log-normal and as trips distances are positive
            scale=2100,  # median of the distribution
            # size=num_trips,  # however many rides you want
        )
    )


# # visualize the random distribution
# samples = [get_random_route_length() for _ in range(10000)]
# plt.hist(samples, bins=100, edgecolor="black")
# plt.title("Distribution of Random Route Lengths")
# plt.xlabel("Route Length")
# plt.ylabel("Frequency")
# plt.show()


# %%
# Generate the routes by walk towards a random bearing


# Returns a route that walks towards a given bearing, stopping when the length threshold is reached
# Returns None if the threshold is not reached
def walkTowardsBearing(
    G: MultiDiGraph, start_node: int, lengthThreshold: float, bearing: float
) -> Tuple[Optional[List[int]], float]:
    bearing = np.float64(bearing) % 360  # normalize bearing to [0, 360)
    n = start_node
    visited_nodes = set([n])
    route = [n]
    routeLength = 0.0
    while routeLength < lengthThreshold:
        successors = set(v for v in G.successors(n))
        successors = list(successors - visited_nodes)
        if len(successors) == 0:
            logging.debug(
                f"No successors for node {n}, stopping the route with length {routeLength} where expected length was {lengthThreshold}"
            )
            return None, 0

        bearing_nodes = []
        for next_n in successors:
            bearing_value = cast(Any, G[n][next_n]).get(0, {}).get("bearing", -1)

            if bearing_value == -1:
                logging.debug(f"Node without bearing: {next_n}")
                continue

            bearing_nodes.append((next_n, bearing_value))

        if not bearing_nodes:
            logging.debug(
                f"No successors with bearing for node {n}, stopping the route"
            )
            return None, 0

        next_n = n
        min_bearing = 999
        for candidate, candidate_bearing in bearing_nodes:
            diff = abs(bearing - candidate_bearing) % 360
            candidate_minimal_diff = min(diff, 360 - diff)

            if candidate_minimal_diff < min_bearing:
                min_bearing = candidate_minimal_diff
                next_n = candidate

        edge = G[n][next_n].get(0, None)  # pyright: ignore
        if not edge:
            logging.error(f"Edge from {n} to {next_n} not found, skipping")
            return None, 0

        edge_length = edge.get("length", 0)
        if edge_length == 0:
            logging.error(f"Edge from {n} to {next_n} has length 0, stopping the route")
            return None, 0

        routeLength += edge_length
        route.append(next_n)
        n = next_n
    return route, routeLength


def generate_route(G: MultiDiGraph) -> Tuple[List[int], float]:
    requested_trip_length = float(get_random_route_length())
    while True:
        start_node = random.choice(list(G.nodes))
        # route, routeLength = randomWalk(G, start_node, requested_trip_length)
        route, routeLength = walkTowardsBearing(
            G, start_node, requested_trip_length, random.random() * 360
        )
        if route == None:
            continue  # no route found satisfying the threshold, skip this iteration

        return route, routeLength


# %%
# Generate events from a route
def create_events_from_route(
    G: MultiDiGraph, route: List[int], start_ts: float, end_ts: float
) -> pd.DataFrame:
    # go from one node to another until finishing the route, creating events along the way
    trip_id = uuid.uuid4()  # uuid
    u = route[0]  # start node
    u_idx = 0  # index of node u in route
    prev_u = -1  # none
    timestamp_at_u = datetime.now()  # declares the variable
    events: List[dict] = list()
    while True:
        if prev_u == -1:
            # Create a random timestamp within specified time range
            random_ts = random.uniform(start_ts, end_ts)
            random_dt = datetime.fromtimestamp(random_ts, tz=berlin_tz)
            timestamp_at_u = random_dt
        else:
            # calculate the timestamp at the next node based on the travel time
            travel_time = float(G[prev_u][u][0]["travel_time"])  # pyright: ignore
            timestamp_at_u = timestamp_at_u + timedelta(seconds=travel_time)

        # create an event
        event = {
            "event_id": uuid.uuid4(),  # unique event ID
            "trip_id": trip_id,
            "timestamp": timestamp_at_u.isoformat(),
            "latitude": G.nodes[u]["y"],
            "longitude": G.nodes[u]["x"],
        }

        events.append(event)  # append event to the list

        # if u not the last node in the route, move to the next node
        if u_idx < len(route) - 1:
            prev_u = u
            u_idx += 1
            u = route[u_idx]
        else:
            break

    df = pd.DataFrame(events)
    df.set_index("event_id", inplace=True)
    return df


# %%
# Create the trips in a loop

# Settings
num_trips = 10000000
berlin_tz = ZoneInfo("Europe/Berlin")
start_ts = datetime(2020, 1, 1, tzinfo=berlin_tz).timestamp()
end_ts = datetime(2025, 12, 31, tzinfo=berlin_tz).timestamp()
filename = "output/escooter_trips_simple.csv"
# NOTE: optional, only needed for visualization of routes
# routes = list()

logging.info(f"Simulating {num_trips} trips")
start_time = time.time()
generated_events = 0
for i in range(num_trips):
    if i % 1000 == 0:
        logging.info(
            f"created {i}/{num_trips} routes and {generated_events} events in {time.time() - start_time:.2f}s"
        )

    route, routeLength = generate_route(G)
    events = create_events_from_route(G, route, start_ts, end_ts)

    # NOTE: optional, only needed for visualization of routes
    # routes.append(route)

    generated_events += len(events)

    logging.debug(
        f"Created trip from node:{route[0]} to node:{route[-1]} with length {routeLength}, and {len(events)}"
    )

    if i == 0:
        # Create the file with the headers for the first trip
        events.to_csv(filename, mode="w", header=True)
    else:
        # For the following trips, append to the file without headers
        events.to_csv(filename, mode="a", header=False)

end_time = time.time()
logging.info(f"Simulated {num_trips} trips in {end_time - start_time:.2f} seconds")

# (Optional) Visualize routes
# fig, ax = ox.plot_graph_routes(G, routes[:100], route_color="blue")
# fig.show()
# fig.savefig("route_example.png")
