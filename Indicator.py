class Indicator:
    def __init__(self, name, period, gradient=False, sign=False):
        self.name = name
        self.period = period
        self.gradient = gradient
        self.sign = sign

    def to_string(self):
        parts = [self.name, self.period]
        if self.gradient:
            parts += 'dx'

        if self.sign:
            parts += '+/-'

        return '_'.join(parts)