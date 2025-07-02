def log(message):
    with open("training.log", "a", encoding="utf-8") as f:
        f.write(message + "\n")
