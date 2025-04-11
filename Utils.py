class RingBuffer:
    def __init__(self, size):
        """
        Initialize the ring buffer with a fixed size.
        :param size: Maximum number of frames the buffer can hold.
        """
        self.size = size
        self.buffer = [None] * size
        self.index = 0
        self.is_full = False

    def add_frame(self, frame):
        """
        Add a frame to the ring buffer.
        :param frame: The frame to add (numpy array).
        """
        self.buffer[self.index] = frame
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.is_full = True

    def get_frames(self):
        """
        Retrieve all frames in the buffer in the correct order.
        :return: List of frames in the buffer.
        """
        if not self.is_full:
            return self.buffer[:self.index]
        return self.buffer[self.index:] + self.buffer[:self.index]