import unittest
from unittest.mock import patch
from sony_wifi_yolo import parse_args


class ShowcaseDefaultsTests(unittest.TestCase):
    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=AssertionError('Default startup must not prompt'))
    def test_testing_starts_case2_without_prompt(self, *_):
        args = parse_args(['--testing', '--haptics-port', '/dev/test'])
        self.assertEqual((args.haptics_case, args.pulse_gap), (2, .3))
        self.assertTrue(args.cup_target)
        self.assertFalse(args.manual_target)

    def test_port_only_uses_final_profile(self):
        args = parse_args(['--haptics-port', 'auto'])
        self.assertEqual((args.haptics_case, args.pulse_gap), (2, .3))
        self.assertTrue(args.cup_target)

    def test_alternate_cases_and_gap_remain_available(self):
        for case in (1, 2, 3):
            args = parse_args(['--haptics-port', 'auto', '--haptics-case', str(case), '--pulse-gap', '.5'])
            self.assertEqual((args.haptics_case, args.pulse_gap), (case, .5))

    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=['2', '0.1'])
    def test_explicit_interactive_selection(self, *_):
        args = parse_args(['--choose-case'])
        self.assertTrue(args.testing)
        self.assertEqual((args.haptics_port, args.haptics_case, args.pulse_gap), ('auto', 2, .1))

    def test_without_hardware_does_not_enable_serial(self):
        args = parse_args(['--cup-target'])
        self.assertIsNone(args.haptics_port)
        self.assertIsNone(args.haptics_case)


if __name__ == '__main__':
    unittest.main()
