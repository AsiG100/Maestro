class Logger():
    def __init__(self) -> None:
        logs = []
    
    def log(self, msg: str):
        self.logs.append(msg)
        print(msg)
