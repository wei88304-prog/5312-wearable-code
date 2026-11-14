import time
import unittest
from sony_haptics import SerialHaptics, ZERO
from test_sony_haptics import Esp32Peer, state


class CasesTests(unittest.TestCase):
    def test_band_boundaries_and_normalization(self):
        for x, band in ((20, 1), (63, 1), (64, 2), (159, 2), (160, 3)):
            with Esp32Peer() as peer:
                h = SerialHaptics(peer.port, case=1)
                try:
                    result = h.update(state(x, 0), 640, 480, observed_at=time.monotonic())
                    self.assertEqual(result['band'], band)
                    self.assertIn(f',1,{band},300,0,1,0,0', peer.commands[-1])
                    result = h.update(state(x, 0), 640, 480, observed_at=time.monotonic()-1)
                    self.assertEqual(result['band'], 0)
                    self.assertFalse(any(result['directions'].values()))
                finally:
                    h.close()

    def test_switch_sends_stop_and_validates_gap(self):
        with Esp32Peer() as peer:
            h = SerialHaptics(peer.port)
            h.configure(2, 100)
            self.assertEqual(peer.frames[-1], ZERO)
            self.assertEqual((h.case, h.gap_ms), (2, 100))
            with self.assertRaises(ValueError):
                h.configure(2, 200)
            h.close()

    def test_case2_gap_and_case3_profile(self):
        with Esp32Peer() as peer:
            h = SerialHaptics(peer.port, case=2, gap_ms=500)
            try:
                h.update(state(100, 0), 640, 480, observed_at=time.monotonic())
                self.assertIn(',2,2,500,0,1,0,0', peer.commands[-1])
                h.configure(3, 100)
                h.update(state(200, 0), 640, 480, observed_at=time.monotonic())
                self.assertIn(',3,3,100,0,1,0,0', peer.commands[-1])
            finally:
                h.close()
