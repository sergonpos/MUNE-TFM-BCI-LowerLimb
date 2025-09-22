# shim_lsl.py
from types import SimpleNamespace
import numpy as np, time

class LSLWorkerShim:
    def __init__(self, lsl_reader, stream_name="Signal_generator"):
        self.reader = lsl_reader
        self.receiver = SimpleNamespace(
            l_cha=self._labels_or_synthetic(),
            fs=float(getattr(self.reader, "fs", 256.0) or 256.0),
            name=stream_name,
        )

    def _labels_or_synthetic(self):
        try:
            labels = self.reader.get_lsl_labels()
        except Exception:
            labels = None
        if labels and all(labels):
            return labels
        n = getattr(self.reader, "ch_count", None)
        if n is None and getattr(self.reader, "inlet", None):
            try: n = self.reader.inlet.info().channel_count()
            except Exception: n = None
        n = n or 8
        return [f"Ch{i+1}" for i in range(n)]

    def get_data(self):
        epoch = self.reader.get_epoch()
        if epoch is None:
            C = len(self.receiver.l_cha)
            return np.empty((0,), dtype=np.float64), np.empty((0, C), dtype=np.float64)
        fs = self.receiver.fs
        n  = epoch.shape[0]
        t1 = time.perf_counter()
        times = (t1 - (np.arange(n, dtype=np.float64)[::-1] + 1) / fs)
        return times, epoch

