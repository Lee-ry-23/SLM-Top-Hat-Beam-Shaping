class Logger:
    def __init__(self):
        self.eval_history = []
        self.iter_history = []
        self.iter_points = []
        self.last_value = None

    def log_evaluation(self, value):
        self.last_value = value
        self.eval_history.append(value)

    def log_iteration(self, xk):
        if self.last_value is not None:
            self.iter_history.append(self.last_value)
        self.iter_points.append(xk.copy())

    def print_last(self):
        if self.last_value is not None:
            print(f"Loss: {self.last_value:.6e}")
