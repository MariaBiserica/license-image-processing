def transform_score(original_score):
    """
    Transform a score from a 0-100 range (0=good, 100=bad) to a 1-5 range (1=bad, 5=good).

    Parameters:
    original_score (int): Score in the 0-100 range.

    Returns:
    float: Transformed score in the 1-5 range.
    """

    # Inverse the original score since a lower score means better in the original scale,
    # but a higher score means better in the target scale.
    inverted_score = 100 - original_score

    # Linearly map the inverted score from 0-100 range to 1-5 range.
    # Formula: Target = ((Original - Original_min) / (Original_max - Original_min)) * (Target_max - Target_min) + Target_min
    transformed_score = ((inverted_score - 0) / (100 - 0)) * (5 - 1) + 1

    return transformed_score

# Example usage
example_score = 75
transformed_example_score = transform_score(example_score)
print(transformed_example_score)
