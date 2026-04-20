import json

# Chemin vers ton fichier
json_path = "/home/abrimont/partage/Survey_Perf/datasets/msvc.json"

# Charger le fichier JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Extraire toutes les questions
questions = [item["question"] for item in data if "question" in item]

# Afficher les questions uniques
unique_questions = sorted(set(questions))
print("Questions uniques trouvées :\n")
for q in unique_questions:
    print("-", q)

print(f"\nTotal unique questions: {len(unique_questions)}")
