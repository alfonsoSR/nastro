from nastro.types.time import JulianDay, DeltaJD, CalendarDate
import nastro.constants as nc
import numpy as np
import pytest


@pytest.mark.parametrize(
    "input, output",
    [  # Ideal cases [Double]
        ((2451545.5, 0.0), (2451545.5, 0.0)),
        ((2451545.5, None), (2451545.5, 0.0)),
        ((2451545.5, 0.5), (2451545.5, 0.5)),
        # Just integral [Double]
        ((2451545.76, None), (2451545.5, 0.26)),
        ((2451545.0, None), (2451544.5, 0.5)),
        ((2451545.2, None), (2451544.5, 0.7)),
        # Fractional part [Double]
        ((2451545.5, 1.5), (2451546.5, 0.5)),
        ((2451545.5, 2.7), (2451547.5, 0.7)),
        ((2451545.5, 3.2), (2451548.5, 0.2)),
        ((2451545.5, 0.9), (2451545.5, 0.9)),
        ((2451545.5, -2.8), (2451542.5, 0.2)),
        # Merge all previous cases into one vectorized case
        (
            (
                np.array(
                    [
                        2451545.5,
                        2451545.5,
                        2451545.76,
                        2451545.0,
                        2451545.2,
                        2451545.5,
                        2451545.5,
                        2451545.5,
                        2451545.5,
                        2451545.5,
                    ]
                ),
                np.array([0.0, 0.5, 0.0, 0.0, 0.0, 1.5, 2.7, 3.2, 0.9, -2.8]),
            ),
            (
                np.array(
                    [
                        2451545.5,
                        2451545.5,
                        2451545.5,
                        2451544.5,
                        2451544.5,
                        2451546.5,
                        2451547.5,
                        2451548.5,
                        2451545.5,
                        2451542.5,
                    ]
                ),
                np.array([0.0, 0.5, 0.26, 0.5, 0.7, 0.5, 0.7, 0.2, 0.9, 0.2]),
            ),
        ),
    ],
)
def test_splitter(input, output) -> None:

    jd = JulianDay(*input)
    if isinstance(jd.day, np.ndarray):
        assert not jd.scalar
    else:
        assert jd.scalar

    assert np.all(np.isclose(jd.day, output[0], atol=1e-15, rtol=0.0))

    if jd.size > 1:
        atol = 1e-15 * np.ones(jd.size)
        for idx, val in enumerate(input[0]):
            if (val + 0.5) % 1 != 0:
                atol[idx] = float(f"1e-{15 - int(np.log10(val))}")
            if not np.isclose(jd.time[idx], output[1][idx], atol=atol[idx], rtol=0.0):
                raise AssertionError(f"Error: {jd.time[idx] - output[1][idx]:.16e}")
    else:
        atol = 1e-15
        if (input[0] + 0.5) % 1 != 0:
            atol = float(f"1e-{15 - int(np.log10(input[0]) + 1)}")
        if not np.isclose(jd.time, output[1], atol=atol, rtol=0.0):
            raise AssertionError(f"Error: {jd.time - output[1]:.16e}")

    return None


@pytest.mark.parametrize(
    "ref, input, output",
    [
        (JulianDay(2451545.5, 0.7), 1.0, JulianDay(2451544.5, 0.7)),
        (JulianDay(2451545.5, 0.7), 1.3, JulianDay(2451544.5, 0.4)),
        (JulianDay(2451545.5, 0.7), 0.8, JulianDay(2451544.5, 0.9)),
        (
            JulianDay(
                np.array([2451545.5, 2451545.5, 2451545.5]),
                np.array([0.7, 0.7, 0.7]),
            ),
            np.array([1.0, 1.3, 0.8]),
            JulianDay(
                np.array([2451544.5, 2451544.5, 2451544.5]),
                np.array([0.7, 0.4, 0.9]),
            ),
        ),
        (JulianDay(2451545.5, 0.7), JulianDay(2451545.5, 0.7), DeltaJD(0.0, 0.0)),
        (JulianDay(2451545.5, 0.7), JulianDay(2451544.5, 0.4), DeltaJD(1.0, 0.3)),
        (JulianDay(2451545.5, 0.7), JulianDay(2451545.5, 0.9), DeltaJD(0.0, -0.2)),
    ],
)
def test_subtraction(ref, input, output) -> None:

    epoch = ref
    result = epoch - input
    assert np.allclose(result.day, output.day, atol=1e-15, rtol=0.0)
    assert np.allclose(result.time, output.time, atol=1e-15, rtol=0.0)

    return None


def test_split_public() -> None:

    input = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 21.5, 22.5, 55.5, 56.5, 57.5])
    epochs = JulianDay(input)
    idx_list, passes = epochs.split(10.0)
    assert idx_list == [0, 5, 7]
    assert np.allclose(passes[0].day, input[:5], atol=1e-15, rtol=0.0)
    assert np.allclose(passes[1].day, input[5:7], atol=1e-15, rtol=0.0)
    assert np.allclose(passes[2].day, input[7:], atol=1e-15, rtol=0.0)

    return None


def test_Date() -> None:
    """Test basic functionality of Date class. [Robust]"""

    date = CalendarDate(2019, 6, 22, 16, 12, 56)
    assert date.year == 2019
    assert date.month == 6
    assert date.day == 22
    assert date.hour == 16
    assert date.minute == 12
    assert date.second == 56
    assert date.as_list() == [2019, 6, 22, 16, 12, 56]

    return None


def test_date2jd() -> None:
    """Test conversion from Date to JulianDate. [Robust]"""

    date = CalendarDate(2019, 6, 22, 16, 12, 56)
    expected_int = 2458656.5
    expected_frac = 0.6756481481481481
    expected_jd = expected_int + expected_frac
    expected_mint = 58656.5
    expected_mfrac = 0.1756481481481481
    expected_mjd = expected_mint + expected_mfrac
    expected_output = JulianDay(2458656.5, 0.6756481481481481)
    converted_jd = date.as_jd()

    assert np.allclose(converted_jd.day, expected_int, atol=1e-15, rtol=0.0)
    assert np.allclose(converted_jd.time, expected_frac, atol=1e-15, rtol=0.0)
    assert np.allclose(converted_jd.jd, expected_jd, atol=1e-15, rtol=0.0)
    assert np.allclose(converted_jd.mday, expected_mint, atol=1e-15, rtol=0.0)
    assert np.allclose(converted_jd.mtime, expected_mfrac, atol=1e-15, rtol=0.0)
    assert np.allclose(converted_jd.mjd, expected_mjd, atol=1e-15, rtol=0.0)
    assert converted_jd == expected_output

    return None


def test_jd2date() -> None:
    """Test conversion from JulianDate to Date. [Robust]"""

    julian_date = JulianDay(2458656.5, 0.6756712962962963)
    expected_date = CalendarDate(2019, 6, 22, 16, 12, 58)
    converted_date = julian_date.as_calendar()

    assert converted_date == expected_date

    # Time series of jds
    julian_date_series = JulianDay(
        np.array([2458656.5, 2458656.5]),
        np.array([0.6756712962962963, 0.6756712962962963]),
    )
    expected_ouput = [expected_date, expected_date]
    converted_output = julian_date_series.as_calendar()
    assert isinstance(converted_output, list)
    for expected, converted in zip(expected_ouput, converted_output):
        assert converted == expected

    return None
