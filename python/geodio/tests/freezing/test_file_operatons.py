
import numpy as np
from geodio.core.freezing import file_operations

if __name__ == "__main__":
    # Create sample weights
    weights = [np.random.rand(3, 3), np.random.rand(2, 2, 2)]
    print("INITIAL:", weights)

    # Write weights to file
    file_operations.write_weights_to_file("weights.bin", weights)

    # Test loading each weight by index
    for i in range(len(weights)):
        loaded_weight = file_operations.load_weight_from_file("weights.bin", i)
        print(f"LOADED WEIGHT {i}:")
        print(loaded_weight)
        # Verify the loaded weight matches the original
        np.testing.assert_array_equal(loaded_weight, weights[i])

    print("All tests passed!")
