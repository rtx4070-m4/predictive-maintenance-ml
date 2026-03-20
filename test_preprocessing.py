
from src.preprocessing import generate_synthetic_data

def test_data_generation():
    df = generate_synthetic_data(100)
    assert not df.empty
