realitivity = 3

# Grenzen des Suchbereichs (i muss zwischen 0 und 1 liegen)
untere_grenze = 0.0
obere_grenze = 1.0

# Wir wiederholen das Spiel 50 Mal – das reicht für extreme Genauigkeit
for _ in range(50):
    i = (untere_grenze + obere_grenze) / 2  # Genau die Mitte raten
    j = i * realitivity
    summe = i + j

    if summe > 1.0:
        # Zu groß! Also liegt das richtige i in der linken/unteren Hälfte
        obere_grenze = i
    else:
        # Zu klein! Also liegt das richtige i in der rechten/oberen Hälfte
        untere_grenze = i

print(f"i = {i}, j = {j}")
# Ausgabe: i = 0.25, j = 0.75
