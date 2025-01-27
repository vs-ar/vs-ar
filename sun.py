import numpy as np

def calculate_sunlit_length(buildings, sun_source):
    def is_edge_sunlit(edge_start, edge_end, source):
        # Compute vectors
        edge_vector = np.array(edge_end) - np.array(edge_start)
        source_vector = np.array(edge_start) - np.array(source)
        # Compute cross product and check if sunlight hits the edge
        cross_product = np.cross(edge_vector, source_vector)
        return cross_product > 0

    total_length = 0.0

    for building in buildings:
        num_points = len(building)
        for i in range(num_points):
            edge_start = building[i]
            edge_end = building[(i + 1) % num_points]  # Next point, cyclically
            if is_edge_sunlit(edge_start, edge_end, sun_source):
                # Add the length of the sunlit edge
                length = np.linalg.norm(np.array(edge_end) - np.array(edge_start))
                total_length += length

    return round(total_length, 2)

# Example Usage
buildings_coords = [
    [[4, 0], [4, -5], [7, -5], [7, 0]],
    [[0.4, -2], [0.4, -5], [2.5, -5], [2.5, -2]]
]
sun_source = [-3.5, 1]

result = calculate_sunlit_length(buildings_coords, sun_source)
print(result)
