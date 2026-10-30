from geo import distance_meters, locations, seconds


def test_distances_and_duration_use_returned_values_and_standard_units():
    a={'latitude':55.7,'longitude':37.6}
    assert distance_meters(a,a) == 0
    assert 1100 < distance_meters(a,dict(a,latitude=55.71)) < 1120
    assert seconds('1231s') == 1231 and seconds('1.5s') == 1.5
    assert seconds('unknown') is None
    assert list(locations({'places':[{'location':a}]})) == [a]
