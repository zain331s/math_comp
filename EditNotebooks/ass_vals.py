def assign_values(strings):
    """
    Assigns unique values to unique strings in the list and replaces the strings with their assigned values.
    Strings that are the same get the same value.

    Args:
        strings (list): List of strings.

    Returns:
        list: A list with strings replaced by their assigned values.
    """
    value_map = {}
    current_value = 0

    for string in strings:
        if string not in value_map:
            value_map[string] = current_value
            current_value += 1

    return [value_map[string] for string in strings],value_map

# Example usage
random_strings = ["apple", "banana", "apple", "cherry", "banana", "date"]
assigned_values,val_map= assign_values(random_strings)
print(assigned_values)
print(val_map)