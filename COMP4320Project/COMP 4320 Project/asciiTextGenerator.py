# Simple python script to generate a 90 Kb ASCII file

text = "Intro to Networks: Tears "

target_size = 90000  # 90 Kb in bytes
repetitions = target_size // len(text)

final_content = text * repetitions

with open("TextFile", "w") as file:
    file.write(final_content)