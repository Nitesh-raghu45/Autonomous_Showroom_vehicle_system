from app.sql_engine import _format_results


def test_format_results_does_not_convert_lakh_values_twice():
    result = _format_results([{"avg_price_lakhs": 28.600689655}], "average price of suvs")

    assert result == "**Avg Price Lakhs**: ₹28.60 L"