import torch

class data_prefetcher(object):
    def __init__(self, loader):
        super(data_prefetcher, self).__init__()

        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()

        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        _input, target = self.next_input, self.next_target
        self.preload()

        return _input, target
