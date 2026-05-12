import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../src"
        )
    )
)

from query_expander import MockGenerativeModel


def test_query_expansion():

    model = MockGenerativeModel()

    query = "How does the system handle peak load?"

    expanded = model.expand_query(query)

    assert expanded != query

    assert "autoscaling" in expanded