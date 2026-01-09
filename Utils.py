class RingBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [None] * size
        self.index = 0
        self.is_full = False

    def add_frame(self, frame):
        self.buffer[self.index] = frame
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.is_full = True

    def get_frames(self):
        if not self.is_full:
            return self.buffer[:self.index]
        return self.buffer[self.index:] + self.buffer[:self.index]