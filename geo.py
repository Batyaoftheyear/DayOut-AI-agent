"""Derived checks over coordinates and durations actually returned by MCP."""
import math
import re


def distance_meters(a, b):
    lat1, lat2 = math.radians(a['latitude']), math.radians(b['latitude'])
    dy = lat2-lat1
    dx = math.radians(b['longitude']-a['longitude'])
    value = math.sin(dy/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dx/2)**2
    return 2*6371000*math.asin(min(1, math.sqrt(value)))


def locations(payload):
    if isinstance(payload, dict):
        point = payload.get('location')
        if isinstance(point, dict) and all(isinstance(point.get(key), (int,float)) for key in ('latitude','longitude')):
            yield point
        for item in payload.values():
            yield from locations(item)
    elif isinstance(payload, list):
        for item in payload:
            yield from locations(item)


def seconds(value):
    if isinstance(value, str) and re.fullmatch(r'\d+(?:\.\d+)?s', value):
        return float(value[:-1])
    return None
