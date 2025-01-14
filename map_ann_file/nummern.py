import random

# Generiere 150 eindeutige Zahlen zwischen 1 und 4628
unique_numbers = random.sample(range(1, 6018), 20)

#unique_numbers  = list(range(1, 4630))


# Sortiere die Liste
sorted_numbers = sorted(unique_numbers)

# Ausgabe der Zahlen als kommaseparierte Liste
print(', '.join(map(str, sorted_numbers)))
