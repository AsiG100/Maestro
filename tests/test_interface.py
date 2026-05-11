import unittest

from ui.interface import ChatUserInterface


class FakeEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.launch_called = False

    def launch(self):
        self.launch_called = True


class ChatUserInterfaceTest(unittest.TestCase):
    def test_constructor_wires_chat_function_into_interface(self):
        ui = ChatUserInterface(interface_engine=FakeEngine)

        self.assertIsInstance(ui.interface_engine, FakeEngine)
        self.assertIs(ui.interface_engine.kwargs["fn"].__self__, ui)
        self.assertIs(ui.interface_engine.kwargs["fn"].__func__, ChatUserInterface.chat)

    def test_chat_formats_message_and_history(self):
        ui = ChatUserInterface(interface_engine=FakeEngine)

        result = ui.chat(
            "Hello",
            history=[{"role": "assistant", "content": "Hi"}],
        )

        self.assertEqual(
            result,
            "Hello{'role': 'assistant', 'content': 'Hi'}from base user interface!",
        )

    def test_launch_delegates_to_interface(self):
        ui = ChatUserInterface(interface_engine=FakeEngine)

        ui.launch()

        self.assertTrue(ui.interface_engine.launch_called)


if __name__ == "__main__":
    unittest.main()
