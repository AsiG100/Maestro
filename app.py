from ui.interface import ChatUserInterface
import gradio as gr

def app():
    interface = ChatUserInterface(
        interface_engine=gr.ChatInterface,
        user_type="kid",
    )
    interface.launch()


if __name__ == "__main__":
    app()
