def log(logpath, message):
    with open(logpath, "w", encoding="utf-8") as f:
        f.write(message + "\n")
