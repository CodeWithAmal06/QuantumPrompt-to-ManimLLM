import importlib.util
import sys
import types
import unittest
from pathlib import Path

# Stub optional runtime dependencies so importing main.py does not fail in a minimal test environment.
for module_name in ["torch", "qwen_vl_utils", "rich.console", "rich.panel", "rich.prompt", "IPython.display"]:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)

sys.modules["rich.console"].Console = object
sys.modules["rich.panel"].Panel = lambda *args, **kwargs: None
sys.modules["rich.prompt"].Prompt = types.SimpleNamespace(ask=lambda *args, **kwargs: "")
sys.modules["IPython.display"].Video = None
sys.modules["IPython.display"].display = lambda *args, **kwargs: None
sys.modules["qwen_vl_utils"].process_vision_info = lambda messages: ([], [])

module_path = Path(__file__).resolve().parents[1] / "main.py"
spec = importlib.util.spec_from_file_location("main_module", module_path)
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)


class TestVLMParsing(unittest.TestCase):
    def test_extracts_json_from_fenced_response(self):
        raw_output = "```json\n{\"valid\": false, \"feedback\": \"Needs a clearer motion transition\"}\n```"

        result = main_module.parse_vlm_output(raw_output)

        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result["valid"])
        self.assertEqual(result["feedback"], "Needs a clearer motion transition")


if __name__ == "__main__":
    unittest.main()
