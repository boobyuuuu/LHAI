def log(logpath, message):
    with open(logpath, "a", encoding="utf-8") as f:
        f.write(message + "\n")
